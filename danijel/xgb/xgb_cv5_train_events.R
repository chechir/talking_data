
rm(list = ls(all = TRUE))
options(scipen = 999)

## load libraries
library(data.table)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(lubridate)
library(xgboost)
library(Matrix)

## load functions
source('mlogloss.R')

## load data
load('./input/train.RData')
load('./input/device.RData')
load('./input/events.RData')
load('./input/app_events.RData')
load('./input/app_labels.RData')
load('./input/label_categories.RData')

## remove column
app_events[, is_installed := NULL]
gc()

## select fold10 and rename it to fold
train$fold10 = NULL
colnames(train)[5] = 'fold'

## merging data
train = merge(train, device, by = 'device_id', all.x = TRUE)
app_labels = merge(app_labels, label_categories, by = 'label_id', all.x = TRUE)
app_events = merge(app_events, events[,.(event_id, device_id)], by = 'event_id', all.x = TRUE)

events$longitude = ifelse(events$longitude == 0, NA, events$longitude)
events$latitude = ifelse(events$latitude == 0, NA, events$latitude)

## has_events devices
train$has_events = ifelse(train$device_id %in% app_events$device_id, 1, 0)

## create numeric features
train$group = as.numeric(factor(train$group)) - 1
train$gender = ifelse(train$gender == 'F', 0, 1)
train$phone_brand = as.numeric(factor(train$phone_brand))
train$device_model = as.numeric(factor(train$device_model))

## age group variable
train$age_group = ifelse(train$gender == 0,
                         as.numeric(cut(train$age, c(-Inf, 23, 26, 28, 32, 42, Inf))),
                         as.numeric(cut(train$age, c(-Inf, 22, 26, 28, 31, 38, Inf)))) - 1

## select only devices with events
train = train[has_events == 1,]
gc()

## all unique devices
all_devices = sort(unique(train$device_id))

## select only events of has_events devices
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
idx.row = rep(1:nrow(train), ncategories)
idx.col = as.numeric(factor(unlist(cat_names)))
count = unlist(cat_count)

cat_feat = sparseMatrix(i = idx.row, j = idx.col, x = count)
colnames(cat_feat) = all_categories

## app features
app_names = lapply(apps_per_device, function(l) names(l))

napps = sapply(apps_per_device, function(l) length(l))
idx.row = rep(1:nrow(train), napps)
idx.col = as.numeric(factor(unlist(app_names)))
count = unlist(apps_per_device)

app_feat = sparseMatrix(i = idx.row, j = idx.col, x = count)
colnames(app_feat) = all_installed_apps

## event features
events$weekday = wday(events$timestamp)
events$hour = hour(events$timestamp)

event_feat = events[device_id %in% train$device_id] %>%
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

#---------------------------------------------------------------------------------------
# select features
#---------------------------------------------------------------------------------------

number_of_installs_per_app = sort(table(unlist(app_names)), decreasing = TRUE)
sel_apps = names(number_of_installs_per_app)[1:1000]

sel_categories = names(which(colSums(cat_feat) > 25))

#---------------------------------------------------------------------------------------
# training data
#---------------------------------------------------------------------------------------

xtrain = as.matrix(cbind(cat_feat[,sel_categories], app_feat[,sel_apps], as.matrix(event_feat),
                         train$phone_brand, train$device_model))

## cv params
nfolds = max(train$fold)
nbags = 5

#---------------------------------------------------------------------------------
# cross validation gender for devices with events
#---------------------------------------------------------------------------------

p_gender = matrix(nrow = nrow(train), ncol = 2)

set.seed(111)
for(i in 1:nfolds){

  ## train / validation index
  inTr = which(train$fold != i)
  inTe = which(train$fold == i)

  ## response
  y = train$gender

  ## xgb data
  dtr = xgb.DMatrix(xtrain[inTr,], label = y[inTr], missing = NA)
  dval = xgb.DMatrix(xtrain[inTe,], label = y[inTe], missing = NA)

  ## parameter set
  param = list(booster            = 'gbtree',
               objective          = 'reg:logistic',
               eval_metric        = 'logloss',
               learning_rate      = 0.025,
               max_depth          = 6,
               subsample          = 0.8,
               colsample_bytree   = 0.5,
               colsample_bylevel  = 0.5)

  p = numeric(length(inTe))
  for(j in 1:nbags){
    ## train model
    bst_gender = xgb.train(params   = param,
                           data     = dtr,
                           nrounds  = 1000)

    ## prediction
    p = p + predict(bst_gender, dval)

    rm(bst_gender)
    gc()
  }

  ## average predictions
  p = p/nbags

  prob = cbind(1-p, p)
  p_gender[inTe,] = prob

  ## calculate score
  score = mlogloss(y[inTe], prob)
  cat('Gender fold', i, '- Score', round(score,6), '\n')
}

score = mlogloss(y, p_gender)
cat('Gender total score', round(score,6), '\n')

#---------------------------------------------------------------------------------
# cross validation age group for devices with events
#---------------------------------------------------------------------------------

p_age_group = matrix(nrow = nrow(train), ncol = 12)
p_tmp = matrix(nrow = nrow(train), ncol = 6)
gender = train$gender

set.seed(222)
for(i in 1:nfolds){

  ## train / validation index
  inTr = which(train$fold != i)
  inTe = which(train$fold == i)

  ## response
  y = train$age_group

  ## add gender

  ## create modified val data
  idx = rep(inTe, each = 2)
  xval_mod = xtrain[idx,]
  xval_mod = cbind(rep(0:1, length(inTe)), xval_mod)

  ## xgb data
  dtr = xgb.DMatrix(cbind(gender[inTr], xtrain[inTr,]), label = y[inTr], missing = NA)
  dval = xgb.DMatrix(cbind(gender[inTe], xtrain[inTe,]), label = y[inTe], missing = NA)
  dval_mod = xgb.DMatrix(xval_mod, missing = NA)

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

  prob = matrix(0, nrow = length(inTe), ncol = 12)
  prob_tmp = matrix(0, nrow = length(inTe), ncol = 6)
  for(j in 1:nbags){
    ## train model
    bst_age_group = xgb.train(params   = param,
                              data     = dtr,
                              nrounds  = 1060)

    ## prediction
    prob = prob + matrix(predict(bst_age_group, dval_mod), ncol = 12, byrow = TRUE)
    prob_tmp = prob_tmp + matrix(predict(bst_age_group, dval), ncol = 6, byrow = TRUE)

    rm(bst_age_group)
    gc()
  }

  ## average predictions
  prob = prob/nbags
  prob_tmp = prob_tmp/nbags

  p_age_group[inTe,] = prob

  ## calculate score
  score = mlogloss(y[inTe], prob_tmp)
  cat('Age_group fold', i, '- Score', round(score,6), '\n')
}

#---------------------------------------------------------------------------------
# combine predictions
#---------------------------------------------------------------------------------

p_group = cbind(p_age_group[,1:6] / rowSums(p_age_group[,1:6]) * p_gender[,1],
                p_age_group[,7:12] / rowSums(p_age_group[,7:12]) * p_gender[,2])

## fold scores
score = sapply(1:nfolds, function(fold) mlogloss(train$group[train$fold == fold], p_group[train$fold == fold,]))

for(i in 1:nfolds) cat('Fold -', i, 'Score', round(score[i], 6), '\n')

## total score
score = mlogloss(train$group, p_group)
cat('Total score', round(score, 6), '\n')

## save predictions
df = data.frame(cbind(train$device_id, p_group))
colnames(df) = c('device_id', 'F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+', 
                 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+')  

write.csv(df, file = './preds/xgb_cv5_train_events.csv', row.names = FALSE, quote = FALSE)
