
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

## has_events devices
tr_te$has_events = ifelse(tr_te$device_id %in% app_events$device_id, 1, 0)

## create numeric features
tr_te$group = as.numeric(factor(tr_te$group)) - 1
tr_te$gender = ifelse(tr_te$gender == 'F', 0, 1)
tr_te$phone_brand = as.numeric(factor(tr_te$phone_brand))
tr_te$device_model = as.numeric(factor(tr_te$device_model))

## age group variable
tr_te$age_group = ifelse(tr_te$gender == 0,
                         as.numeric(cut(tr_te$age, c(-Inf, 23, 26, 28, 32, 42, Inf))),
                         as.numeric(cut(tr_te$age, c(-Inf, 22, 26, 28, 31, 38, Inf)))) - 1

## select only rows with has_events devices
tr_te_act = tr_te[has_events == 1,]
gc()

## all unique devices
all_devices = sort(unique(tr_te_act$device_id))

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

## app features
app_names = lapply(apps_per_device, function(l) names(l))

napps = sapply(apps_per_device, function(l) length(l))
idx.row = rep(1:nrow(tr_te_act), napps)
idx.col = as.numeric(factor(unlist(app_names)))
count = unlist(apps_per_device)

app_feat = sparseMatrix(i = idx.row, j = idx.col, x = count)
colnames(app_feat) = all_installed_apps

## category features

cat_count = lapply(apps_per_device, function(l) table(unlist(categories_per_app[names(l)])))
cat_names = lapply(cat_count, function(l) names(l))

all_categories = sort(unique(unlist(cat_names)))

ncategories = sapply(cat_names, function(l) length(l))
idx.row = rep(1:nrow(tr_te_act), ncategories)
idx.col = as.numeric(factor(unlist(cat_names)))
count = unlist(cat_count)

cat_feat = sparseMatrix(i = idx.row, j = idx.col, x = count)
colnames(cat_feat) = all_categories

## model/brand features

model = sparse.model.matrix(has_events ~ -1 + factor(device_model), data = tr_te)
brand = sparse.model.matrix(has_events ~ -1 + factor(phone_brand), data = tr_te)

rm(events)
gc()

#---------------------------------------------------------------------------------------
# merge data
#---------------------------------------------------------------------------------------

sel_event = which(tr_te$has_events == 1)
sel_noevent = which(tr_te$has_events == 0)

xtr_te_event = cbind(1, model[sel_event,], brand[sel_event,], cat_feat, app_feat)
xtr_te_noevent = cbind(0, model[sel_noevent,], brand[sel_noevent,],
                       Matrix(0, nrow = length(sel_noevent), ncol = ncol(cat_feat) + ncol(app_feat)))

rownames(xtr_te_event) = sel_event
rownames(xtr_te_noevent) = sel_noevent

xtr_te = rbind(xtr_te_event, xtr_te_noevent)
xtr_te = xtr_te[order(as.numeric(rownames(xtr_te))),]

rm(xtr_te_event, xtr_te_noevent)
gc()

#---------------------------------------------------------------------------------------
# split train/test
#---------------------------------------------------------------------------------------

xtrain = xtr_te[tr_te$is_train == 1,]
xtest = xtr_te[tr_te$is_train == 0,]

#---------------------------------------------------------------------------------------
# select features
#---------------------------------------------------------------------------------------

sel_col = which(colSums(xtrain) > 0)

first_n_for_noevents = length(grep('factor', names(sel_col))) + 1
# 1559

xtrain = xtrain[,sel_col]
xtest = xtest[,sel_col]

rm(cat_feat, app_feat, model, brand)
gc()

#---------------------------------------------------------------------------------------
# transform to binary
#---------------------------------------------------------------------------------------

xtrain[xtrain > 0] = 1
xtest[xtest > 0] = 1

#---------------------------------------------------------------------------------------
# save data
#---------------------------------------------------------------------------------------

sparse2triples = function(m){
  SM = summary(m)
  D1 = m@Dimnames[[1]][SM[,1]]
  D2 = m@Dimnames[[2]][SM[,2]]
  data.frame(row = D1, col = D2, x = m@x)
}

rownames(xtrain) = 1:nrow(xtrain)
colnames(xtrain) = 1:ncol(xtrain)
triples_tr = sparse2triples(xtrain)
triples_tr$row = as.numeric(as.character(triples_tr$row)) - 1
triples_tr$col = as.numeric(as.character(triples_tr$col)) - 1

rownames(xtest) = 1:nrow(xtest)
colnames(xtest) = 1:ncol(xtest)
triples_te = sparse2triples(xtest)
triples_te$row = as.numeric(as.character(triples_te$row)) - 1
triples_te$col = as.numeric(as.character(triples_te$col)) - 1


write.csv(triples_tr, './input/for_py_xtrain_triples_all.csv', row.names = FALSE, quote = FALSE)
write.csv(triples_te, './input/for_py_xtest_triples_all.csv', row.names = FALSE, quote = FALSE)

write.csv(tr_te$group[tr_te$is_train == 1], './input/for_py_group_all.csv', row.names = FALSE, quote = FALSE)
write.csv(tr_te$gender[tr_te$is_train == 1], './input/for_py_gender_all.csv', row.names = FALSE, quote = FALSE)
write.csv(tr_te$age_group[tr_te$is_train == 1], './input/for_py_agegroup_all.csv', row.names = FALSE, quote = FALSE)
write.csv(tr_te$device_id[tr_te$is_train == 1], './input/for_py_train_device_id_all.csv', row.names = FALSE, quote = FALSE)
write.csv(tr_te$device_id[tr_te$is_train == 0], './input/for_py_test_device_id_all.csv', row.names = FALSE, quote = FALSE)

write.csv(tr_te$has_events[tr_te$is_train == 1], './input/for_py_train_has_events.csv', row.names = FALSE, quote = FALSE)
write.csv(tr_te$has_events[tr_te$is_train == 0], './input/for_py_test_has_events.csv', row.names = FALSE, quote = FALSE)

