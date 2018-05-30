
## merge events and no events files for train and test

rm(list = ls(all = TRUE))
options(scipen = 999)

## load libraries
library(data.table)

## read fiels
train_events = fread('./preds/xgb_cv5_train_events.csv', colClasses = c('device_id' = 'character'))
train_noevents = fread('./preds/xgb_cv5_train_noevents.csv', colClasses = c('device_id' = 'character'))
test_events = fread('./preds/xgb_test_events.csv', colClasses = c('device_id' = 'character'))
test_noevents = fread('./preds/xgb_test_noevents.csv', colClasses = c('device_id' = 'character'))

## merge
train_all = rbind(train_events, train_noevents)
train_all = train_all[order(device_id)]

test_all = rbind(test_events, test_noevents)
test_all = test_all[order(device_id)]

## save files
write.csv(train_all, file = './preds/xgb_cv5_train.csv', row.names = FALSE, quote = FALSE)
write.csv(test_all, file = './preds/xgb_test.csv', row.names = FALSE, quote = FALSE)
