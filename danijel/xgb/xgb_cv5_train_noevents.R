
rm(list = ls(all = TRUE))
options(scipen = 999)

## load libraries
library(data.table)
library(dplyr)
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

## select fold10 and rename it to fold
train$fold10 = NULL
colnames(train)[5] = 'fold'

## merging data
train = merge(train, device, by = 'device_id', all.x = TRUE)
app_events = merge(app_events, events[,.(event_id, device_id)], by = 'event_id', all.x = TRUE)

## create numeric features
train$group = as.numeric(factor(train$group)) - 1
train$gender = ifelse(train$gender == 'F', 0, 1)
train$phone_brand = as.numeric(factor(train$phone_brand))
train$device_model = as.numeric(factor(train$device_model))

## new features
train$has_events = ifelse(train$device_id %in% app_events$device_id, 1, 0)

## age group variable
train$age_group = ifelse(train$gender == 0,
                         as.numeric(cut(train$age, c(-Inf, 23, 26, 28, 32, 42, Inf))),
                         as.numeric(cut(train$age, c(-Inf, 22, 26, 28, 31, 38, Inf)))) - 1

## clear workspace
rm(events, app_events)
gc()

## cv params
nfolds = max(train$fold)
nbags = 5

#---------------------------------------------------------------------------------
# cross validation gender for devices without events
#---------------------------------------------------------------------------------

p_gender = matrix(nrow = nrow(train), ncol = 2)

set.seed(111)
for(i in 1:nfolds){
  
  ## train / validation index
  inTr = which(train$fold != i)
  inTe = which(train$fold == i & train$has_events == 0)
  
  ## features
  features = c('phone_brand', 'device_model', 'has_events')
  
  ## response
  y = train$gender
  
  ## xgb data
  dtr = xgb.DMatrix(as.matrix(train[inTr, features, with = FALSE]), label = y[inTr], missing = NA)
  dval = xgb.DMatrix(as.matrix(train[inTe, features, with = FALSE]), label = y[inTe], missing = NA)
  
  p = numeric(length(inTe))
  for(j in 1:nbags){
    ## parameter set
    param = list(booster            = 'gbtree',
                 objective          = 'reg:logistic',
                 eval_metric        = 'logloss',
                 learning_rate      = 0.05,
                 max_depth          = 2,
                 subsample          = 0.8,
                 colsample_bytree   = 0.8)
    
    ## train model
    bst_tmp = xgb.train(params   = param,
                        data     = dtr,
                        nrounds  = 1100)
    
    param$learning_rate = 0.005
    param$max_depth = 50
    
    bst_gender = xgb.train(params     = param,
                           data       = dtr,
                           nrounds    = 200,
                           xgb_model  = bst_tmp)
    
    ## prediction
    p = p + predict(bst_gender, dval)
    
    rm(bst_tmp, bst_gender)
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

sel = which(train$has_events == 0)
score = mlogloss(y[sel], p_gender[sel,])
cat('Gender total score', round(score,6), '\n')

#---------------------------------------------------------------------------------
# cross validation age group for devices without events
#---------------------------------------------------------------------------------

p_age_group = matrix(nrow = nrow(train), ncol = 12)

set.seed(222)
for(i in 1:nfolds){

  ## train / validation index
  inTr = which(train$fold != i)
  inTe = which(train$fold == i & train$has_events == 0)

  ## features
  features = c('phone_brand', 'device_model', 'has_events', 'gender')
  
  ## response
  y = train$age_group
  
  ## create modified val data
  idx = rep(inTe, each = 2)
  val_mod = train[idx,]
  val_mod$gender = rep(0:1, length(inTe))
  
  ## xgb data
  dtr = xgb.DMatrix(as.matrix(train[inTr, features, with = FALSE]), label = y[inTr], missing = NA)
  dval = xgb.DMatrix(as.matrix(train[inTe, features, with = FALSE]), label = y[inTe], missing = NA)
  dval_mod = xgb.DMatrix(as.matrix(val_mod[, features, with = FALSE]), missing = NA)
  
  prob = matrix(0, nrow = length(inTe), ncol = 12)
  prob_tmp = matrix(0, nrow = length(inTe), ncol = 6)
  for(j in 1:nbags){
    ## parameter set
    param = list(booster            = 'gbtree',
                 objective          = 'multi:softprob',
                 eval_metric        = 'mlogloss',
                 num_class          = 6,
                 learning_rate      = 0.05,
                 max_depth          = 7,
                 subsample          = 0.8,
                 colsample_bytree   = 0.8)
    
    ## train model
    bst_tmp = xgb.train(params   = param,
                        data     = dtr,
                        nrounds  = 100)
    
    param$learning_rate = 0.005
    param$max_depth = 50
    
    bst_age_group = xgb.train(params     = param,
                              data       = dtr,
                              nrounds    = 130,
                              xgb_model  = bst_tmp)
    
    ## prediction
    prob = prob + matrix(predict(bst_age_group, dval_mod), ncol = 12, byrow = TRUE)
    prob_tmp = prob_tmp + matrix(predict(bst_age_group, dval), ncol = 6, byrow = TRUE)
    
    rm(bst_tmp, bst_age_group)
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
score = sapply(1:nfolds, function(fold) mlogloss(train$group[train$has_events == 0 & train$fold == fold],
                                         p_group[train$has_events == 0 & train$fold == fold,]))

for(i in 1:nfolds) cat('Fold -', i, 'Score', round(score[i], 6), '\n')

## total score
sel = which(train$has_events == 0)
score = mlogloss(train$group[sel], p_group[sel,])
cat('Total score', round(score, 6), '\n')

# Total score 2.391261 

## save predictions
df = data.frame(cbind(train$device_id, p_group))
colnames(df) = c('device_id', 'F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+', 
                 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+')  
df = na.omit(df)

write.csv(df, file = './preds/xgb_cv5_train_noevents.csv', row.names = FALSE, quote = FALSE)

