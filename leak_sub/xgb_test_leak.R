
rm(list = ls(all = TRUE))
options(scipen = 999)


## load libraries
library(data.table)
library(dplyr)
library(ggplot2)
library(lubridate)
library(xgboost)

## load functions
source('mlogloss.R')

## load data
load('./leak_sub/input/train_leak.RData')
load('./leak_sub/input/test_leak.RData')
load('./danijel/input/device.RData')
load('./danijel/input/events.RData')
load('./danijel/input/app_events.RData')

## merge train and test
train$is_train = 1
train$fold = NULL
test$gender = NA
test$age = NA
test$group = NA
test$is_train = 0
tr_te = rbind(train, test)

## merging data
tr_te = merge(tr_te, device, by = 'device_id', all.x = TRUE)
app_events = merge(app_events, events[,.(event_id, device_id)], by = 'event_id', all.x = TRUE)

## create numeric features
tr_te$group = as.numeric(factor(tr_te$group)) - 1
tr_te$gender = ifelse(tr_te$gender == 'F', 0, 1)
tr_te$phone_brand = as.numeric(factor(tr_te$phone_brand))
tr_te$device_model = as.numeric(factor(tr_te$device_model))

## new features
tr_te$has_events = ifelse(tr_te$device_id %in% app_events$device_id, 1, 0)

## age group variable
tr_te$age_group = ifelse(tr_te$gender == 0,
                         as.numeric(cut(tr_te$age, c(-Inf, 23, 26, 28, 32, 42, Inf))),
                         as.numeric(cut(tr_te$age, c(-Inf, 22, 26, 28, 31, 38, Inf)))) - 1

## split train and test data
train = tr_te[is_train == 1,]
test = tr_te[is_train == 0,]

rm(tr_te, events, app_events)
gc()

#---------------------------------------------------------------------------------
# xgb model - gender - logistic regression
#---------------------------------------------------------------------------------

nbags = 10
p_gender = matrix(0, nrow = nrow(test), ncol = 2)

## features
features = c('phone_brand', 'device_model', 'has_events', 'row')

## xgb data
dtr = xgb.DMatrix(as.matrix(train[, features, with = FALSE]), label = train$gender, missing = NA)

set.seed(111)
for(i in 1:nbags){
  cat('Iter', i, '- gender\n')
  
  ## parameter set
  param = list(booster            = 'gbtree',
               objective          = 'reg:logistic',
               learning_rate      = 0.05,
               max_depth          = 2,
               subsample          = 0.8,
               colsample_bytree   = 0.8)
  
  ## train model
  bst_tmp = xgb.train(params                 = param,
                      data                   = dtr,
                      nrounds                = 1100)
  
  param$learning_rate = 0.005
  param$max_depth = 50
  
  ## train model
  bst_gender = xgb.train(params                 = param,
                         data                   = dtr,
                         nrounds                = 300,
                         xgb_model              = bst_tmp)
  
  p = predict(bst_gender, as.matrix(test[, features, with = FALSE]))
  p_gender = p_gender + cbind(1-p, p)
  
  rm(bst_tmp, bst_gender)
  gc()
}

p_gender = p_gender/nbags

#---------------------------------------------------------------------------------
# xgb model - age group - multinomial regression
#---------------------------------------------------------------------------------

nbags = 10
p_age_group = matrix(0, nrow = nrow(test), ncol = 12)

idx = rep(1:nrow(test), each = 2)
test_mod = test[idx,]
test_mod$gender = rep(0:1, nrow(test))

## features
features = c('phone_brand', 'device_model', 'has_events', 'gender', 'row')

## xgb data
dtr = xgb.DMatrix(as.matrix(train[, features, with = FALSE]), label = train$age_group, missing = NA)

set.seed(222)
for(i in 1:nbags){
  cat('Iter', i, '- age group\n')
  
  ## parameter set
  param = list(booster            = 'gbtree',
               objective          = 'multi:softprob',
               num_class          = 6,
               learning_rate      = 0.05,
               max_depth          = 7,
               subsample          = 0.8,
               colsample_bytree   = 0.8)
  
  ## train model
  bst_tmp = xgb.train(params                 = param,
                      data                   = dtr,
                      nrounds                = 100)
  
  param$learning_rate = 0.005
  param$max_depth = 50
  
  ## train model
  bst_age_group = xgb.train(params                 = param,
                            data                   = dtr,
                            nrounds                = 300,
                            xgb_model              = bst_tmp)
  
  ## prediction
  p_age_group = p_age_group + matrix(predict(bst_age_group, as.matrix(test_mod[, features, with = FALSE])), 
                                     ncol = 12, byrow = TRUE)
  
  rm(bst_tmp, bst_age_group)
  gc()
}

p_age_group = p_age_group/nbags

#---------------------------------------------------------------------------------
# combine no leak submission with xgb leak
#---------------------------------------------------------------------------------

p_group = cbind(p_age_group[,1:6] / rowSums(p_age_group[,1:6]) * p_gender[,1],
                p_age_group[,7:12] / rowSums(p_age_group[,7:12]) * p_gender[,2])

df = cbind(data.frame(test$device_id), p_group)

colnames(df) = c('device_id', 'F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+', 
                 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+')  

best = fread('./no_leak_sub/best_submission_no_leak.csv', colClasses = c('device_id' = 'character'))

sel_noevents = which(test$has_events == 0)

ids_noevents = test$device_id[sel_noevents]
sel_events = which(!(best$device_id %in% ids_noevents))

sub_combined = rbind(df[sel_noevents,], best[sel_events,])
write.csv(sub_combined, file = './leak_sub/best_submission_no_leak_with_xgb_leak.csv', row.names = FALSE, quote = FALSE)

# #---------------------------------------------------------------------------------
# # save xgb leak predictions
# #---------------------------------------------------------------------------------
# 
# write.csv(df, file = './leak_sub/xgb_test_leak.csv', row.names = FALSE, quote = FALSE)

