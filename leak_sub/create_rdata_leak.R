
rm(list = ls(all = TRUE))
options(scipen = 999)

## load libraries
library(data.table)
library(dplyr)

## read data
train = fread('./danijel/input/gender_age_train.csv', colClasses = c('device_id' = 'character'))
test = fread('./danijel/input/gender_age_test.csv', colClasses = c('device_id' = 'character'))

train$row = 1:nrow(train) / nrow(train)
test$row = 1:nrow(test) / nrow(test)

folds5 = fread('./danijel/folds/folds_5.csv', colClasses = c('device_id' = 'character'))

train = merge(train, folds5, by = 'device_id', all.x = TRUE)
train = train[order(device_id)]
test = test[order(device_id)]

## save data as rdata
save(train, file = './leak_sub/input/train_leak.RData')
save(test, file = './leak_sub/input/test_leak.RData')


