
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
tr_te$brand_model_pair = as.numeric(factor(paste(tr_te$phone_brand, tr_te$device_model, sep = ' - ')))

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

## reorder train and test set
train = train[order(row)]
test = test[order(row)]

# train$age_gender_brand_model_tuple = as.numeric(factor(paste(train$age, train$gender, train$brand_model_pair, sep = ' - ')))

runs = rle(train$brand_model_pair)
brand_models = runs$values
train$pair_id = rep(1:length(runs$lengths), runs$lengths)
train$nruns = rep(runs$lengths, runs$lengths)


runs = rle(test$brand_model_pair)
brand_models = runs$values
test$pair_id = rep(1:length(runs$lengths), runs$lengths)
test$nruns = rep(runs$lengths, runs$lengths)


n = c(4, 3, 2)
m_all = NULL
probabilities = c(0.88, 0.74, 0.425)
thresholds = c(200, 150, 100)


for(i in 1:3){
  
  if(i == 1){
    train_sub = train[nruns >= n[i],]
    test_sub = test[nruns >= n[i],]
  }
  
  if(i > 1){
    train_sub = train[nruns == n[i],]
    test_sub = test[nruns >= n[i],]
  }
  
  ave_row_train = sapply(split(train_sub$row, train_sub$pair_id), function(l) mean(l))
  brand_model_train = sapply(split(train_sub$brand_model, train_sub$pair_id), function(l) unique(l))
  
  ave_row_test = sapply(split(test_sub$row, test_sub$pair_id), function(l) mean(l))
  brand_model_test = sapply(split(test_sub$brand_model, test_sub$pair_id), function(l) unique(l))
  
  
  step_size = train$row[2] - train$row[1]
  m = NULL
  
  for(j in 1:length(ave_row_train)){
    
    pair_id_sel_train = as.numeric(names(brand_model_train)[j])
    
    sel = which(brand_model_test == brand_model_train[j])
    ave_rows_sel = ave_row_test[sel]
    pair_id_sel_test = as.numeric(names(ave_row_test)[sel])
    
    if(length(sel) > 0){
      
      dist = abs(ave_rows_sel - ave_row_train[j]) / step_size
      sel_id = which.min(dist)
      if(dist[sel_id] <= thresholds[i]){
        m = rbind(m, c(pair_id_sel_train, pair_id_sel_test[sel_id]))
      }
    }
  }
  
  ## exclude duplicated merges
  exclude = m[duplicated(m[,2]),2]
  sel = which(!(m[,2] %in% exclude))
  m = m[sel,] 
  
  if(i == 1) m_all = m
  if(i > 1){
    sel = which(!(m[,2] %in% m_all[,2]))
    m = m[sel,]
    m_all = rbind(m_all, m)
  }
  
  pred = NULL
  dev = NULL
  
  p = probabilities[i]
  
  for(k in 1:nrow(m)){
    group = unique(train$group[train$pair_id == m[k,1]]) + 1
    te_sel = test[pair_id == m[k,2],]
    
    if(length(group) == 1){
      pred_tmp = matrix((1-p) / 11, nrow = nrow(te_sel), ncol = 12)
      pred_tmp[,group] = p
      dev = c(dev, te_sel$device_id)
      pred = rbind(pred, pred_tmp)
    }
  }
  
  if(i == 1) df = cbind(data.frame(dev), pred)
  if(i > 1) df = rbind(df, cbind(data.frame(dev), pred))
  
}

colnames(df) = c('device_id', 'F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+', 
                 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+')  

#---------------------------------------------------------------------------------
# combine id match leak with no leak best sub combined with xgb leak
#---------------------------------------------------------------------------------

best = fread('./leak_sub/best_submission_no_leak_with_xgb_leak.csv', colClasses = c('device_id' = 'character'))

## thresholding
preds = best[, 2:13, with = FALSE]
preds[preds <= 0.001] = 0.001
preds = preds / rowSums(preds)
best = cbind(best[, 1, with = FALSE], preds)

## combining predictions
sel = which(!(best$device_id %in% df$device_id))
sub_new = rbind(df, best[sel,])

## save submission
write.csv(sub_new, file = './leak_sub/best_submission_with_leak.csv', row.names = FALSE, quote = FALSE)


