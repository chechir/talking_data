rm(list=ls());gc()
# install.packages("dtplyr")
# install.packages("Matrix")
# install.packages("data.table")
# install.packages("caret")
# install.packages("glmnet")
# install.packages("Metrics")
# install.packages("bit64")
library(dtplyr); library(Matrix); library(data.table)
library(caret); library(glmnet);  library(Metrics); library(bit64)
myseed=718

# actual: integer vector with truth labels, values range from 0 to n - 1 classes
# pred_m: predicted probs: column 1 => label 0, column 2 => label 1 and so on
mlogloss = function(actual, pred_m, eps = 1e-15){

  if(max(actual) >= ncol(pred_m) || min(actual) < 0){
    stop(cat('True labels should range from 0 to', ncol(pred_m) - 1, '\n'))
  }

  pred_m[pred_m > 1 - eps] = 1 - eps
  pred_m[pred_m < eps] = eps
  pred_m = t(apply(pred_m, 1, function(r)r/sum(r)))
  actual_m = matrix(0, nrow = nrow(pred_m), ncol = ncol(pred_m))
  actual_m[matrix(c(1:nrow(pred_m), actual + 1), ncol = 2)] = 1
  -sum(actual_m * log(pred_m))/nrow(pred_m)
}

set.seed(myseed)

# library(doParallel)
# registerDoParallel(4)

label_train <- fread("./input/gender_age_train.csv",
                     colClasses=c("character","character",
                                  "integer","character"))
label_test <- fread("./input/gender_age_test.csv",
                    colClasses=c("character"))


#Setea test = a train
label_test$gender <- label_test$age <- label_test$group <- NA

#crea super
label <- rbind(label_train,label_test)

#une a nivel de super mejor:
train_more<-readRDS('rds/trainx2_20160728')
test_more<-readRDS('rds/testx2_20160728')

super_more=rbind(train_more, test_more)
super_more$device_id<-as.character(super_more$device_id)

#order both
super_more<-super_more[order(super_more$device_id),]
label<-label[order(label$device_id),]

sum(super_more$device_id !=label$device_id) ## Must be zero!

#label$with_event<-super_more$with_event

fillNA <- function(x) ifelse(is.na(x), -1, x)
cutQuantile <- function(x, n = 40) cut(x, c(-Inf, unique(quantile(x, 1:(n - 1)/ n)), Inf))

for(f in c("class", "inform", "unknown", "health", "game",
           "date_diff", "babi", "bank", "high", "low", "servic",
           "X0", "X1", "X2", "comic", "mean_lat", "car", "financi", "mean_lon",
           "travel", "shop", "fund", "sport", "share", "educ", "hotel", "napps")) {
  super_more[,f]<-fillNA(super_more[,f])
  label[,f]=as.character(cutQuantile(super_more[,f]))
}


for(i in c('phone_brand', 'device_model')){
  if(!is.numeric(super_more[,i])){
    freq = data.frame(table(super_more[,i]))
    freq = freq[order(freq$Freq, decreasing = TRUE),]

    super_more[,i] = as.numeric(match(super_more[,i], freq$Var1))
  }
}

super_more$device_model<-fillNA(super_more$device_model)
label$pop_model=as.character(cutQuantile(super_more$device_model))

super_more$phone_brand<-fillNA(super_more$phone_brand)
label$pop_brand=as.character(cutQuantile(super_more$phone_brand))

label$with_events=!is.na(super_more$installed)
table(label$with_events)
#colnames(train_more)

##Continue this great script
setkey(label,device_id)
rm(label_test,label_train, super_more);gc()

#carga brands
brand <- fread("./input/phone_brand_device_model.csv",
               colClasses=c("character","character","character"))
setkey(brand,device_id)
#saca duplicados
brand0 <- unique(brand,by=NULL)
brand0 <- brand0[sample(nrow(brand0)),]
brand2 <- brand0[-which(duplicated(brand0$device_id)),]
label1 <- merge(label,brand2,by="device_id",all.x=T)
rm(brand,brand0,brand2);gc()


##Hasta acá sólo ha creado super y agregado las brands -----*

# apps
events <- fread("./input/events.csv",
                colClasses=c("character","character","character",
                             "numeric","numeric"))
setkeyv(events,c("device_id","event_id"))
event_app <- fread("./input/app_events.csv",
                   colClasses=rep("character",4))
setkey(event_app,event_id)

#desduplica eventos
events <- unique(events[,list(device_id,event_id)],by=NULL)

#paste(unique(event_app$app_id),collapse=",") #prueba
#paste(unique(c('a', 'b', 'c')), collapse=",")
#list(x = cars[,1], y = cars[,2], z=cars[,2])  #list crea una lista con los args

str(event_app)
event_apps <- event_app[,list(apps=paste(unique(app_id),collapse=",")),by="event_id"]

#Esto es lo mismo! pero muuuucho mas lento:
#library(dplyr)
#ea<-group_by(event_app[1:10000,], event_id) %>% summarise(apps=paste(unique(app_id), collapse=";"))


device_event_apps <- merge(events,event_apps,by="event_id")
rm(events,event_app,event_apps);gc()

## Crea esta función simplemenmte ra sacar lo unicos
f_split_paste <- function(z){paste(unique(unlist(strsplit(z,","))),collapse=",")}
device_apps <- device_event_apps[,list(apps=f_split_paste(apps)),by="device_id"]
###Explicación con manzanas de esta última línea:

# #primero crea una lista por cada row con apps:
# strsplit(device_event_apps$apps[1:5],',')
#
# #Luego pone todas las apps juntas.. o_O. Por cada device_id (por el group by de después):
# unlist(strsplit(device_event_apps$apps[1:5],','))
#
# #y luego sava los unique
# unique(unlist(strsplit(device_event_apps$apps[1:5],',')))
#
# #Luego aplica todo con un group_by por device:- Acá es lo mismo pero sin llamar a una funcón-
#device_apps5 <- device_event_apps[,list(apps=paste(unique(unlist(strsplit(apps,','))), collapse=",")),by="device_id"]
#str(device_apps5)

##Recordatorio:
#las apps quedaron de manera de que cada registro tiene todos los códigos de apps separados por coma.
device_apps$apps[3]

rm(device_event_apps,f_split_paste);gc()

#Ahora este wey vectoriza lo que está separado por comas
tmp <- strsplit(device_apps$apps,",")


device_apps <- data.table(device_id=rep(device_apps$device_id,
                                        times=sapply(tmp,length)),  # n apps distintas
                          app_id=unlist(tmp)) #unlist tmp will be the same as nrow of device_apps!

# dummy
d1 <- label1[,list(device_id,phone_brand)]
label1$phone_brand <- NULL

dim(label1)
d2 <- label1[,list(device_id,device_model)]
label1$device_model <- NULL

d3 <- device_apps
rm(device_apps)

d4 <- label1[,list(device_id,date_diff)]
d5 <- label1[,list(device_id,babi)]
d6 <- label1[,list(device_id,bank)]
d7 <- label1[,list(device_id,high)]
d8 <- label1[,list(device_id,low)]
d9 <- label1[,list(device_id,servic)]
d10 <- label1[,list(device_id,X0)]
d11 <- label1[,list(device_id,X1)]
d12 <- label1[,list(device_id,comic)]
d13 <- label1[,list(device_id,mean_lat)]
d14 <- label1[,list(device_id,car)]
d15 <- label1[,list(device_id,financi)]
d16 <- label1[,list(device_id,mean_lon)]
d17 <- label1[,list(device_id,pop_model)]
d18 <- label1[,list(device_id,pop_brand)]
d19 <- label1[,list(device_id,class)]
d20 <- label1[,list(device_id,inform)]
d21 <- label1[,list(device_id,unknown)]
d22 <- label1[,list(device_id,health)]
d23 <- label1[,list(device_id,game)]
d24 <- label1[,list(device_id,travel)]
d25 <- label1[,list(device_id,shop)]
d26 <- label1[,list(device_id,fund)]
d27 <- label1[,list(device_id,sport)]
d28 <- label1[,list(device_id,share)]
d29 <- label1[,list(device_id,educ)]
d30 <- label1[,list(device_id,hotel)]
d31 <- label1[,list(device_id,napps)]


d1[,phone_brand:=paste0("phone_brand:",phone_brand)]
d2[,device_model:=paste0("device_model:",device_model)]
d3[,app_id:=paste0("app_id:",app_id)]
d4[,date_diff:=paste0("date_diff:",date_diff)]
d5[,babi:=paste0("babi:",babi)]
d6[,bank:=paste0("bank:",bank)]
d7[,high:=paste0("high:",high)]
d8[,low:=paste0("low:",low)]
d9[,servic:=paste0("servic:",servic)]
d10[,X0:=paste0("X0:",X0)]
d11[,X1:=paste0("X1:",X1)]
d12[,comic:=paste0("comic:",comic)]
d13[,mean_lat:=paste0("mean_lat:",mean_lat)]
d14[,car:=paste0("car:",car)]
d15[,financi:=paste0("financi:",financi)]
d16[,mean_lon:=paste0("mean_lon:",mean_lon)]
d17[,pop_model:=paste0("pop_model:",pop_model)]
d18[,pop_brand:=paste0("pop_brand:",pop_brand)]
d19[,class:=paste0("class:",class)]
d20[,inform:=paste0("inform:",inform)]
d21[,unknown:=paste0("unknown:",unknown)]
d22[,health:=paste0("health:",health)]
d23[,game:=paste0("game:",game)]
d24[,travel:=paste0("travel:",travel)]
d25[,shop:=paste0("shop:",shop)]
d26[,fund:=paste0("fund:",fund)]
d27[,sport:=paste0("sport:",sport)]
d28[,share:=paste0("share:",share)]
d29[,educ:=paste0("educ:",educ)]
d30[,hotel:=paste0("hotel:",hotel)]
d31[,napps:=paste0("napps:",napps)]

#"class", "inform", "unknown", "health", "game"
names(d1) <- names(d2) <- names(d3) <- names(d4) <- names(d5) <- names(d6) <- c("device_id","feature_name")
names(d7) <- names(d8) <- names(d9) <- names(d10) <- names(d11) <- c("device_id","feature_name")
names(d12) <- names(d13) <- names(d14) <- names(d15) <- c("device_id","feature_name")
names(d16) <- names(d17) <- names(d18) <- c("device_id","feature_name")
names(d19) <- names(d20) <- names(d21) <- names(d22) <- names(d23) <- c("device_id","feature_name")
names(d24) <- names(d25) <- names(d26) <- names(d27) <- names(d28) <- c("device_id","feature_name")
names(d29) <- names(d30) <- names(d31) <- c("device_id","feature_name")

dd <- rbind(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31)
rm(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31);gc()

ii <- unique(dd$device_id)
jj <- unique(dd$feature_name)
id_i <- match(dd$device_id,ii)
id_j <- match(dd$feature_name,jj)
id_ij <- cbind(id_i,id_j)

dim(id_ij)

M <- Matrix(0,nrow=length(ii),ncol=length(jj),
            dimnames=list(ii,jj),sparse=T)
dim(M)
M[id_ij] <- 1


rm(ii,jj,id_i,id_j,id_ij,dd);gc()

#filter devices ids that were in events, etc, but not in train:
x <- M[rownames(M) %in% label1$device_id,]


# id <- label1$device_id[match(label1$device_id,rownames(x))]
# y <- label1$group[match(label1$device_id,rownames(x))]

#the same but if not ordered it was better:
id <- label1$device_id[match(rownames(x),label1$device_id)]
y <- label1$group[match(rownames(x),label1$device_id)]

y_age<-label1$age[match(rownames(x),label1$device_id)]
y_gender<-label1$gender[match(rownames(x),label1$device_id)]


with_events=label1$with_events
table(with_events)

rm(M,label1)

# level reduction
x_train <- x[!is.na(y),]
tmp_cnt_train <- colSums(x_train)
x <- x[,tmp_cnt_train>8 & tmp_cnt_train<nrow(x_train)]
dim(x)
rm(x_train)

(group_name <- sort(na.omit(unique(y))))
idx_train <- which(!is.na(y))
idx_test <- which(is.na(y))
train_data <- x[idx_train,]
test_data <- x[idx_test,]
train_label <- match(y[idx_train],group_name)-1
test_label <- match(y[idx_test],group_name)-1

#############################################################
# Modeling
#############################################################

set.seed(myseed)

train_data <- x[idx_train,]
test_data <- x[idx_test,]

### Predictions repository for train
n.preds <- 12
preds <- matrix(0, nrow = nrow(train_data), ncol = n.preds)
predsTest <- matrix(0, nrow = nrow(test_data), ncol = n.preds)
colnames(preds)=colnames(predsTest)=1:n.preds; 

### n Fold using file from Danijel
folds<-fread('./input/folds_5.csv')
n.fold=length(unique(folds$fold))
fold=folds$fold
nbags=5

#check if folds and train_data have the same order:
sum(folds$device_id!=rownames(train_data))#must be zero!

#Dim reduction
dim(train_data)
tmp_cnt_train <- colSums(train_data)

Yvals=as.factor(y[idx_train])

##Only brand and model data - NO EVENTS
features=grep('(brand)|(model)|(row)',colnames(train_data), value=T)

for (fold.id in 1:n.fold) {
  #fold.id=2
  cat("Starting fold ", fold.id, "\n")
  train.id <- fold.id != fold
  valid.id <- fold.id == fold
  train.id.ne<- fold.id != fold & !with_events[idx_train]
  train.id.we<- fold.id != fold & with_events[idx_train]
  valid.id.ne <- fold.id == fold & !with_events[idx_train]
  valid.id.we <- fold.id == fold & with_events[idx_train]
  #length(train.id)+length(valid.id)
  print(Sys.time())
  # GLMNET #####################################
  ##Only brand and model data - NO EVENTS
  print(Sys.time())
  trLog = glmnet(x=train_data[train.id,features],
                 y=Yvals[train.id], family = "multinomial", alpha=0.45)#0.8141108 (0.5)

  predGlm=predict(trLog, train_data[valid.id.ne,features], type="response")[,,31]
  preds[valid.id.ne,1:12]=predGlm
  cat("loglossGlm", fold.id, mlogloss(as.numeric(Yvals[valid.id.ne])-1, predGlm))

  ##aLL data - WITH EVENTS
  trLog = glmnet(x=train_data[train.id,],
                 y=Yvals[train.id], family = "multinomial", alpha=0.45)#0.8141108 (0.5)

  predGlm=predict(trLog, train_data[valid.id.we,], type="response")[,,32]
  preds[valid.id.we,1:12]=predGlm
  cat("loglossGlm", fold.id, mlogloss(as.numeric(Yvals[valid.id.we])-1, predGlm))

  #oVERALL sCORE
  score_val=mlogloss(train_label[valid.id],  preds[valid.id,1:12])
  cat("Overall for fold", fold.id, score_val, "\n")
}



#Overall no events:
score_val=mlogloss(train_label[!with_events[idx_train]],  preds[!with_events[idx_train],1:12])
cat("Overall no events:", fold.id, score_val, "\n")

#Overall WITH events:
score_val=mlogloss(train_label[with_events[idx_train]],  preds[with_events[idx_train],1:12])
cat("Overall WITH events:", fold.id, score_val, "\n")

#Final CV score:
score_val=mlogloss(train_label,  preds[,1:12])
cat("Overall", fold.id, score_val, "\n")

#########################################
# Predicting on the test set
#########################################

test.id.we=with_events[idx_test]
test.id.ne=!with_events[idx_test]

#Without Events
# GLMNET #####################################

print(Sys.time())
trLog = glmnet(x=train_data[,features],
               y=Yvals, family = "multinomial", alpha=0.45)#0.8141108 (0.5)


predGlm=predict(trLog, test_data[test.id.ne,features], type="response")[,,31]
predsTest[test.id.ne,1:12]=predGlm

##aLL data - WITH EVENTS
trLog = glmnet(x=train_data,
               y=Yvals, family = "multinomial", alpha=0.45)#0.8141108 (0.5)


predGlm=predict(trLog, test_data[test.id.we,], type="response")[,,32]
predsTest[test.id.we,1:12]=predGlm


#########################################
# Generate prediction files
#########################################

train_pred=cbind(id=id[idx_train], as.data.frame(preds[,1:12]))
test_pred=cbind(id=id[idx_test], as.data.frame(predsTest[,1:12]))

colnames(train_pred)[1:13]=c("device_id",group_name)
colnames(test_pred)[1:13]=c("device_id",group_name)

write.csv(train_pred, file = 'preds/cv5_shityGlmnet_1hot_d1-d31_train.R.csv', row.names = FALSE, quote = FALSE)
write.csv(test_pred, file = 'preds/cv5_shityGlmnet_1hot_d1-d31_test.R.csv', row.names = FALSE, quote = FALSE)

