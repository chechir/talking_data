
rm(list=ls()); gc()

library(data.table); library(dplyr); library(stringr); library(quanteda); library(bit64)
options(scipen=99)
set.seed(7)

#############
# Loading data
#############

train <- fread('./input/gender_age_train.csv')
test <- fread('./input/gender_age_test.csv')

group<-train$group
train<-select(train, device_id)

## Create train + test
super<-rbind(train, test)

events <- fread('./input/events.csv')
dim(events)

## Reducing df to only train devices
events<-inner_join(events, super, by="device_id")
dim(events)

events$hour = as.numeric(substr(events$timestamp, 12, 13))
events$timestamp<-as.Date(events$timestamp)

app_events <- fread('./input/app_events.csv')

## Reducing df to only train devices
app_events <- inner_join(app_events, unique(dplyr::select(events, event_id)), by="event_id")
dim(app_events)

gc()
labls <- fread('./input/label_categories.csv')
app_labls <-fread('./input/app_labels.csv')

#using only apps that appears in the train set
tmp<-data.table(app_id=unique(app_events$app_id))
app_labls<-inner_join(app_labls, tmp, by="app_id")

rm(tmp, train, test);gc()
gc()

#############
# Calculating bag of labels
#############

cor_lab <- corpus(labls$category)
dfm_lab <- dfm(cor_lab, removeTwitter=TRUE, verbose=FALSE, stem = TRUE, ignoredFeatures=stopwords("english"))

## Using only top 15 labels in terms of frequency
words<-names(topfeatures(dfm_lab, 20))

labls<-cbind(labls, as.data.frame(dfm_lab[,words]))

labls<-left_join(labls, app_labls, by="label_id")
labls$label_id<- labls$category<- NULL


labls<- group_by(labls, app_id) %>%
    summarise(unknown=sum(unknown),
              game=sum(game),
              travel=sum(travel),
              shop=sum(shop),
              car=sum(car),
              servic=sum(servic),
              babi=sum(babi),
              comic=sum(comic),
              fund=sum(fund),
              bank=sum(bank),
              high=sum( high),
              low=sum(low),
              sport=sum(educ),
              share=sum(share),
              educ=sum(educ),
              class=sum(class),
              inform=sum(inform),
              financi=sum(financi),
              hotel=sum(hotel),
              health=sum(health),
              nlab=n())
labls<-as.data.frame(unique(labls))

for(w in c(words, "nlab")){
    labls[,w]=ifelse(labls[,w]>0, 1, 0) #replacing counts by binary value
}

rm(app_labls, cor_lab, dfm_lab); gc()

#############
# Merging
#############

detail<-left_join(app_events, as.data.table(labls), by="app_id")

## Collapsing at event lvl:
event_calc<- group_by(detail, event_id) %>%
    summarise(installed=sum(is_installed, na.rm=T),
              active=sum(is_active, na.rm=T),
              napps=n_distinct(app_id),
              unknown=sum(unknown, na.rm = T),
              game=sum(game, na.rm = T),
              travel=sum(travel, na.rm = T),
              shop=sum(shop, na.rm = T),
              car=sum(car, na.rm = T),
              servic=sum(servic, na.rm = T),
              babi=sum(babi, na.rm = T),
              comic=sum(comic, na.rm = T),
              fund=sum(fund, na.rm = T),
              bank=sum(bank, na.rm = T),
              high=sum(high, na.rm = T),
              low=sum(low, na.rm = T),
              sport=sum(educ, na.rm = T),
              share=sum(share, na.rm = T),
              educ=sum(educ, na.rm = T),
              class=sum(class, na.rm = T),
              inform=sum(inform, na.rm = T),
              financi=sum(financi, na.rm = T),
              hotel=sum(hotel, na.rm = T),
              health=sum(health, na.rm = T),
              nlab=sum(nlab, na.rm = T)
    )

rm(detail, app_events, labls);gc()


#############
# Merging events and collapsing to device_id level
#############

event_calc<-left_join(event_calc, events, by="event_id")

## Summarizing to match device level
device_calc<- group_by(event_calc, device_id) %>%
    summarise(mean_lat=mean(latitude, na.rm=T),
              sd_lat=sd(latitude, na.rm=T),
              mean_lon=mean(longitude, na.rm=T),
              sd_lon=sd(longitude, na.rm=T),
              mean_hour=mean(hour, na.rm=T),
              sd_hour=sd(hour, na.rm=T),
              installed=sum(installed, na.rm=T),
              active=sum(active, na.rm=T),
              min_date=min(timestamp, na.rm=T),
              max_date=max(timestamp, na.rm=T),
              napps=sum(napps),
              nevents=n_distinct(event_id),
              unknown=sum(unknown, na.rm = T),
              game=sum(game, na.rm = T),
              travel=sum(travel, na.rm = T),
              shop=sum(shop, na.rm = T),
              car=sum(car, na.rm = T),
              servic=sum(servic, na.rm = T),
              babi=sum(babi, na.rm = T),
              comic=sum(comic, na.rm = T),
              fund=sum(fund, na.rm = T),
              bank=sum(bank, na.rm = T),
              high=sum( high, na.rm = T),
              low=sum(low, na.rm = T),
              sport=sum(educ, na.rm = T),
              share=sum(share, na.rm = T),
              educ=sum(educ, na.rm = T),
              class=sum(class, na.rm = T),
              inform=sum(inform, na.rm = T),
              financi=sum(financi, na.rm = T),
              hotel=sum(hotel, na.rm = T),
              health=sum(health, na.rm = T),
              nlab=sum(nlab, na.rm=T)
    )

device_calc<- as.data.frame(device_calc)
for(w in words){
    device_calc[,w]<- device_calc[,w]/device_calc$nlab
}

rm(event_calc); gc()

brands <- fread('./input/phone_brand_device_model.csv')

# Brands have duplicated rows!. Let's remove them:
brands<-group_by(brands, device_id) %>%
    summarise(phone_brand=min(phone_brand),
              device_model=min(device_model))


##Creating the super set
super<-left_join(super, brands, by="device_id")
super<-left_join(super, device_calc, by="device_id")

super$with_event<-ifelse(is.na(super$installed), 0, 1)
super$date_diff<-as.numeric(super$max_date - super$min_date)

test <- fread('./input/gender_age_test.csv')
super$group<-c(group, rep('0', nrow(test)))

train<-head(super, sum(super$group!='0'))
test<-tail(super, sum(super$group=='0'))

#############
# Importing external sources
#############

###ADDING PYTHON iNPUT:
train_py<-as.data.frame(fread("rds/train_svd200.csv", header = TRUE))
test_py<-as.data.frame(fread("rds/test_svd200.csv", header = TRUE))

train_py<-dplyr::select(train_py, -V1)
test_py<-dplyr::select(test_py, -V1)

#order both
train<-train[order(train$device_id),]
test<-test[order(test$device_id),]

train_py<-train_py[order(train_py$device_id),]
test_py<-test_py[order(test_py$device_id),]

sum(train_py$device_id != train$device_id) #Must be zero!
sum(test_py$device_id != test$device_id) #Must be zero!

train_py$device_id=NULL
test_py$device_id=NULL

train<-cbind(train, train_py)
test<-cbind(test, test_py)

dim(train)
dim(test)

colnames(train)<-make.names(colnames(train))
colnames(test)<-make.names(colnames(test))


saveRDS(train, 'rds/trainx2_20160728')
saveRDS(test, 'rds/testx2_20160728')
