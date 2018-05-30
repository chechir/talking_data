

## merge events and no events files for train and test

rm(list = ls(all = TRUE))
options(scipen = 999)

## load libraries
library(data.table)

## read fiels
train_events = fread('./preds/keras_2_cv5_train_events.csv', colClasses = c('device_id' = 'character'))
train_noevents = fread('./preds/keras_2_cv5_train_noevents.csv', colClasses = c('device_id' = 'character'))
test_events = fread('./preds/keras_2_test_events.csv', colClasses = c('device_id' = 'character'))
test_noevents = fread('./preds/keras_2_test_noevents.csv', colClasses = c('device_id' = 'character'))

## select rows
sel_train = which(!(train_noevents$device_id %in% train_events$device_id))
sel_test = which(!(test_noevents$device_id %in% test_events$device_id))

## merge
train_all = rbind(train_noevents[sel_train,], train_events)
train_all = train_all[order(device_id)]

test_all = rbind(test_noevents[sel_test,], test_events)
test_all = test_all[order(device_id)]

## save files
write.csv(train_all, file = './preds/keras_2_cv5_train.csv', row.names = FALSE, quote = FALSE)
write.csv(test_all, file = './preds/keras_2_test.csv', row.names = FALSE, quote = FALSE)



