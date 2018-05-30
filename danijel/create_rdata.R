
rm(list = ls(all = TRUE))
options(scipen = 999)

## load libraries
library(data.table)
library(dplyr)

## read data
train = fread('./input/gender_age_train.csv', colClasses = c('device_id' = 'character'))
test = fread('./input/gender_age_test.csv', colClasses = c('device_id' = 'character'))
device = fread('./input/phone_brand_device_model.csv', colClasses = c('device_id' = 'character'),
                      encoding = "UTF-8", stringsAsFactors = FALSE)
device = device[!duplicated(device$device_id),]
events = fread('./input/events.csv', colClasses = c('device_id' = 'character'))
app_events = fread('./input/app_events.csv', colClasses = c('app_id' = 'character'))
app_labels = fread('./input/app_labels.csv', colClasses = c('app_id' = 'character'))
label_categories = fread('./input/label_categories.csv')


folds5 = fread('folds/folds_5.csv', colClasses = c('device_id' = 'character'))
names(folds5)[2] = 'fold5'

folds10 = fread('folds/folds_10.csv', colClasses = c('device_id' = 'character'))
names(folds10)[2] = 'fold10'

train = merge(train, folds5, by = 'device_id', all.x = TRUE)
train = merge(train, folds10, by = 'device_id', all.x = TRUE)
test = test[order(device_id)] 

## save data as rdata
save(train, file = './input/train.RData')
save(test, file = './input/test.RData')
save(device, file = './input/device.RData')
save(events, file = './input/events.RData')
save(app_events, file = './input/app_events.RData')
save(app_labels, file = './input/app_labels.RData')
save(label_categories, file = './input/label_categories.RData')

## clear workspace and quit
rm(list = ls(all = TRUE))
gc()
quit()