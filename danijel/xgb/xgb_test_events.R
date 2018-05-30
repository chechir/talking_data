
rm(list = ls(all = TRUE))
options(scipen = 999)

## load libraries
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(xgboost)
library(Matrix)

## load functions
source('mlogloss.R')
source('nsmallest.R')

## load data
load('./input/train.RData')
load('./input/test.RData')
load('./input/device.RData')
load('./input/events.RData')
load('./input/app_events.RData')
load('./input/app_labels.RData')
load('./input/label_categories.RData')

## merge train and test
train$is_train = 1
train$fold5 = NULL
train$fold10 = NULL
test$gender = NA
test$age = NA
test$group = NA
test$is_train = 0
tr_te = rbind(train, test)

rm(train, test)
gc()

## merging data
tr_te = merge(tr_te, device, by = 'device_id', all.x = TRUE)
app_labels = merge(app_labels, label_categories, by = 'label_id', all.x = TRUE)
app_events = merge(app_events, events[,.(event_id, device_id)], by = 'event_id', all.x = TRUE)

events$longitude = ifelse(events$longitude == 0, NA, events$longitude)
events$latitude = ifelse(events$latitude == 0, NA, events$latitude)

## active devices
tr_te$active = ifelse(tr_te$device_id %in% app_events$device_id, 1, 0)

## create numeric features
tr_te$group = as.numeric(factor(tr_te$group)) - 1
tr_te$gender = ifelse(tr_te$gender == 'F', 0, 1)
tr_te$phone_brand = as.numeric(factor(tr_te$phone_brand))
tr_te$device_model = as.numeric(factor(tr_te$device_model))

## age group variable
tr_te$age_group = ifelse(tr_te$gender == 0,
                         as.numeric(cut(tr_te$age, c(-Inf, 23, 26, 28, 32, 42, Inf))),
                         as.numeric(cut(tr_te$age, c(-Inf, 22, 26, 28, 31, 38, Inf)))) - 1

## select only rows with active devices
tr_te = tr_te[active == 1,]
gc()

## all unique devices
all_devices = sort(unique(tr_te$device_id))

## select only events of active devices
events = events[device_id %in% all_devices,]
app_events = app_events[device_id %in% all_devices,]
gc()

#---------------------------------------------------------------------------------------
# create lists for feature extraction
#---------------------------------------------------------------------------------------

## all ids of installed apps
all_installed_apps = sort(unique(app_events$app_id))

## apps per device
apps_per_device = app_events[order(device_id)] %>%
  with(., split(app_id, device_id)) %>%
  lapply(., function(l) table(l)) 

## categories per app
categories_per_app = app_labels[order(app_id)] %>%
  .[app_id %in% all_installed_apps,] %>%
  with(., split(category, app_id)) %>%
  lapply(., function(l) sort(unique(l)))
gc()

rm(app_events)
gc()

#---------------------------------------------------------------------------------------
# feature extraction
#---------------------------------------------------------------------------------------

## category features

cat_count = lapply(apps_per_device, function(l) table(unlist(categories_per_app[names(l)])))
cat_names = lapply(cat_count, function(l) names(l))

all_categories = sort(unique(unlist(cat_names)))

ncategories = sapply(cat_names, function(l) length(l))
idx.row = rep(1:nrow(tr_te), ncategories)
idx.col = as.numeric(factor(unlist(cat_names)))
count = unlist(cat_count)

cat_feat = sparseMatrix(i = idx.row, j = idx.col, x = count)
colnames(cat_feat) = all_categories

## app features
app_names = lapply(apps_per_device, function(l) names(l))

napps = sapply(apps_per_device, function(l) length(l))
idx.row = rep(1:nrow(tr_te), napps)
idx.col = as.numeric(factor(unlist(app_names)))
count = unlist(apps_per_device)

app_feat = sparseMatrix(i = idx.row, j = idx.col, x = count)
colnames(app_feat) = all_installed_apps

## event features
events$weekday = wday(events$timestamp)
events$hour = hour(events$timestamp)

event_feat = events[device_id %in% tr_te$device_id] %>%
  .[order(device_id)] %>%
  .[, .(nevents = .N, 
        ## location
        mlong = median(longitude, na.rm = TRUE), mlat = median(latitude, na.rm = TRUE),
        ## week
        w1 = sum(weekday == 1), w2 = sum(weekday == 2), w3 = sum(weekday == 3), w4 = sum(weekday == 4),
        w5 = sum(weekday == 5), w6 = sum(weekday == 6), w7 = sum(weekday == 7),
        ## hour
        h1 = sum(hour == 0), h2 = sum(hour == 1), h3 = sum(hour == 2), h4 = sum(hour == 3), h5 = sum(hour == 4),
        h6 = sum(hour == 5), h7 = sum(hour == 6), h8 = sum(hour == 7), h9 = sum(hour == 8), h10 = sum(hour == 9),
        h11 = sum(hour == 10), h12 = sum(hour == 11), h13 = sum(hour == 12), h14 = sum(hour == 13),
        h15 = sum(hour == 14), h16 = sum(hour == 15), h17 = sum(hour == 16), h18 = sum(hour == 17),
        h19 = sum(hour == 18), h20 = sum(hour == 19), h21 = sum(hour == 20), h22 = sum(hour == 21),
        h23 = sum(hour == 22), h24 = sum(hour == 23)),
    by = device_id] 

event_feat[, device_id := NULL]

rm(events)
gc()

#---------------------------------------------------------------------------------------
# train/test index
#---------------------------------------------------------------------------------------

inTr = which(tr_te$is_train == 1)
inTe = which(tr_te$is_train == 0)

#---------------------------------------------------------------------------------------
# select features
#---------------------------------------------------------------------------------------

number_of_installs_per_app = sort(table(unlist(app_names[inTr])), decreasing = TRUE)
sel_apps = names(number_of_installs_per_app)[1:1000]

sel_categories = names(which(colSums(cat_feat[inTr,]) > 25))

#---------------------------------------------------------------------------------------
# create train/test data
#---------------------------------------------------------------------------------------

xtrain = as.matrix(cbind(cat_feat[inTr,sel_categories], app_feat[inTr,sel_apps], as.matrix(event_feat)[inTr,],
                         tr_te$phone_brand[inTr], tr_te$device_model[inTr]))

xtest = as.matrix(cbind(cat_feat[inTe,sel_categories], app_feat[inTe,sel_apps], as.matrix(event_feat)[inTe,],
                        tr_te$phone_brand[inTe], tr_te$device_model[inTe]))


y_gender = tr_te$gender[inTr]
y_age_group = tr_te$age_group[inTr]

id_train = tr_te$device_id[inTr]
id_test = tr_te$device_id[inTe]

rm(tr_te, cat_feat, app_feat, event_feat, all_categories, all_devices, all_installed_apps, app_labels, 
   app_names, apps_per_device, cat_count, cat_names, categories_per_app, count, idx.row, idx.col)
gc()

#---------------------------------------------------------------------------------
# xgb model - gender - logistic regression
#---------------------------------------------------------------------------------

nbags = 10
p_gender = matrix(0, nrow = nrow(xtest), ncol = 2)

## xgb data
dtr = xgb.DMatrix(xtrain, label = y_gender, missing = NA)

## parameter set
param = list(booster            = 'gbtree',
             objective          = 'reg:logistic',
             eval_metric        = 'logloss',
             learning_rate      = 0.025,
             max_depth          = 6,
             subsample          = 0.8,
             colsample_bytree   = 0.5,
             colsample_bylevel  = 0.5)


set.seed(555)
for(i in 1:nbags){
  cat('Iter', i, '- gender\n')
  
  ## train model
  bst_gender = xgb.train(params   = param,
                         data     = dtr,
                         nrounds  = 1000)
  
  ## prediction
  p = predict(bst_gender, xtest)
  p_gender = p_gender + cbind(1-p, p)
  
  rm(bst_gender)
  gc()
}

p_gender = p_gender/nbags

#---------------------------------------------------------------------------------
# xgb model - age group - multinomial
#---------------------------------------------------------------------------------

nbags = 10
p_age_group = matrix(0, nrow = nrow(xtest), ncol = 12)

## create modified test data
idx = rep(1:nrow(xtest), each = 2)
xtest_mod = xtest[idx,]
xtest_mod = cbind(rep(0:1, nrow(xtest)), xtest_mod)
rm(xtest, dtr)
gc()

## xgb data
dtr = xgb.DMatrix(cbind(y_gender, xtrain), label = y_age_group, missing = NA)

## parameter set
param = list(booster            = 'gbtree',
             objective          = 'multi:softprob',
             eval_metric        = 'mlogloss',
             num_class          = 6,
             learning_rate      = 0.025,
             max_depth          = 6,
             subsample          = 0.8,
             colsample_bytree   = 0.5,
             colsample_bylevel  = 0.5)


set.seed(666)
for(i in 1:nbags){
  cat('Iter', i, '- age group\n')
  
  ## train model
  bst_age_group = xgb.train(params   = param,
                            data     = dtr,
                            nrounds  = 1060)
  
  ## prediction
  p_age_group = p_age_group + matrix(predict(bst_age_group, xtest_mod), ncol = 12, byrow = TRUE)
  
  rm(bst_age_group)
  gc()
}

p_age_group = p_age_group/nbags

#---------------------------------------------------------------------------------
# create submission
#---------------------------------------------------------------------------------

p_group = cbind(p_age_group[,1:6] / rowSums(p_age_group[,1:6]) * p_gender[,1],
                p_age_group[,7:12] / rowSums(p_age_group[,7:12]) * p_gender[,2])

## save predictions
df = data.frame(cbind(test$device_id, p_group))
colnames(df) = c('device_id', 'F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+', 
                 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+')  
write.csv(df, file = './preds/xgb_test_events.csv', row.names = FALSE, quote = FALSE)

