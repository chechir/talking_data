library(data.table); library(dplyr); library(stringr); library(bit64); library("nnet")
rm(list=ls())

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

## READ PREDICTION FILES
#train_tools<-readRDS("preds/train_preds_20160731_cv10.rds")[, c("device_id", "y")]
train_tools<-fread("./input/gender_age_train.csv", colClasses=c("character","character",
                                                                "integer","character"))


train_pred1<-fread("../danijel/preds/xgb_cv5_train.csv", colClasses = c(device_id="character"))
test_pred1<-fread("../danijel/preds/xgb_test.csv", colClasses = c(device_id="character"))

train_pred2<-fread("../danijel/preds/keras_2_cv5_train.csv", colClasses = c(device_id="character"))
test_pred2<-fread("../danijel/preds/keras_2_test.csv", colClasses = c(device_id="character"))

train_pred3<-fread("preds/cv5_shityXgb_1hot_d1-d31_bagged_train.R.csv", colClasses = c(device_id="character"))
test_pred3<-fread("preds/cv5_shityXgb_1hot_d1-d31_bagged_test.R.csv", colClasses = c(device_id="character"))

train_pred3b<-fread("preds/cv5_shityXgb_1hot_d1-d31_bagged_seed2_train.R.csv", colClasses = c(device_id="character"))
test_pred3b<-fread("preds/cv5_shityXgb_1hot_d1-d31_bagged_seed2_test.R.csv", colClasses = c(device_id="character"))

#Augmenting bagged for shityXgb:
train_pred3=as.data.frame(train_pred3);test_pred3=as.data.frame(test_pred3)
train_pred3b=as.data.frame(train_pred3b);test_pred3b=as.data.frame(test_pred3b)
for(i in 2:13){
  train_pred3[,i]=(train_pred3[,i]+ train_pred3b[,i])/2
  test_pred3[,i]=(test_pred3[,i]+ test_pred3b[,i])/2
}


train_pred4<-fread("preds/cv5_shityXgbLinear_1hot_d1-d31_bagged_train.R.csv", colClasses = c(device_id="character"))
test_pred4<-fread("preds/cv5_shityXgbLinear_1hot_d1-d31_bagged_test.R.csv", colClasses = c(device_id="character"))

train_pred5<-fread("preds/cv5_shityGlmnet_1hot_d1-d31_train.R.csv", colClasses = c(device_id="character"))
test_pred5<-fread("preds/cv5_shityGlmnet_1hot_d1-d31_test.R.csv", colClasses = c(device_id="character"))

train_pred6<-fread("preds/keras_cv5_2_bagging_split_train.csv", colClasses = c(device_id="character"))
test_pred6<-fread("preds/keras_cv5_2_bagging_split_test.csv", colClasses = c(device_id="character"))    

train_pred7<-fread("../danijel/preds/keras_1_cv5_train.csv", colClasses = c(device_id="character"))
test_pred7<-fread("../danijel/preds/keras_1_test.csv", colClasses = c(device_id="character"))

train_pred8<-fread("preds/keras_cv5_1_bagging_train.csv", colClasses = c(device_id="character"))
test_pred8<-fread("preds/keras_cv5_1_bagging_test.csv", colClasses = c(device_id="character"))

train_pred9<-fread("preds/cv5_shityXgb_2_1hot_d1-d31_bagged_train.R.csv", colClasses = c(device_id="character"))
test_pred9<-fread("preds/cv5_shityXgb_2_1hot_d1-d31_bagged_test.R.csv", colClasses = c(device_id="character"))


train_pred10<-fread("preds/keras_pred_train_bags5_wEvents_allData2_20160824.csv", colClasses = c(device_id="character"))
test_pred10<-fread("preds/keras_pred_test_bags5_wEvents_allData2_20160824.csv", colClasses = c(device_id="character"))

train_pred10b<-fread("preds/keras_pred_train_bags5_wEvents_allData2_20160824_seed1.csv", colClasses = c(device_id="character"))
test_pred10b<-fread("preds/keras_pred_test_bags5_wEvents_allData2_20160824_seed1.csv", colClasses = c(device_id="character"))

train_pred10c<-fread("preds/keras_pred_train_bags5_wEvents_allData2_20160824_seed2.csv", colClasses = c(device_id="character"))
test_pred10c<-fread("preds/keras_pred_test_bags5_wEvents_allData2_20160824_seed2.csv", colClasses = c(device_id="character"))

#Creating bagged train for keras:
train_pred10=as.data.frame(train_pred10)
train_pred10b=as.data.frame(train_pred10b)
train_pred10c=as.data.frame(train_pred10c)
for(i in 14:25){
  train_pred10[,i]=(train_pred10[,i]+ train_pred10b[,i]+ train_pred10c[,i])/3
}
#Creating bagged test for keras:
test_pred10=as.data.frame(test_pred10)
test_pred10b=as.data.frame(test_pred10b)
test_pred10c=as.data.frame(test_pred10c)
for(i in 14:25){
  test_pred10[,i]=(test_pred10[,i]+ test_pred10b[,i]+ test_pred10c[,i])/3
}

train_pred11<-fread("../danijel/preds/keras_3_cv10_train.csv", colClasses = c(device_id="character"))
test_pred11<-fread("../danijel/preds/keras_3_test.csv", colClasses = c(device_id="character"))

train_pred12<-fread("preds/keras_pred_train_bags5_20160819.csv", colClasses = c(device_id="character"))
test_pred12<-fread("preds/keras_pred_test_bags_SVD20_20160819.csv", colClasses = c(device_id="character"))

## ORDER PREDICTION FILES
train_tools=train_tools[order(train_tools$device_id),]

train_pred1=train_pred1[order(train_pred1$device_id),]
test_pred1=test_pred1[order(test_pred1$device_id),]

train_pred2=train_pred2[order(train_pred2$device_id),]
test_pred2=test_pred2[order(test_pred2$device_id),]

train_pred3=train_pred3[order(train_pred3$device_id),]
test_pred3=test_pred3[order(test_pred3$device_id),]

train_pred4=train_pred4[order(train_pred4$device_id),]
test_pred4=test_pred4[order(test_pred4$device_id),]

train_pred5=train_pred5[order(train_pred5$device_id),]
test_pred5=test_pred5[order(test_pred5$device_id),]

train_pred6=train_pred6[order(train_pred6$device_id),]
test_pred6=test_pred6[order(test_pred6$device_id),]

train_pred7=train_pred7[order(train_pred7$device_id),]
test_pred7=test_pred7[order(test_pred7$device_id),]

train_pred8=train_pred8[order(train_pred8$device_id),]
test_pred8=test_pred8[order(test_pred8$device_id),]

train_pred9=train_pred9[order(train_pred9$device_id),]
test_pred9=test_pred9[order(test_pred9$device_id),]

train_pred10=train_pred10[order(train_pred10$device_id),]
test_pred10=test_pred10[order(test_pred10$device_id),]

train_pred11=train_pred11[order(train_pred11$device_id),]
test_pred11=test_pred11[order(test_pred11$device_id),]

train_pred12=train_pred12[order(train_pred12$device_id),]
test_pred12=test_pred12[order(test_pred12$device_id),]

#Small test to see if everything is ok
sum(train_pred1$device_id!=train_tools$device_id)
sum(train_pred3$device_id!=train_pred1$device_id);sum(test_pred3$device_id!=test_pred1$device_id)
sum(train_pred3$device_id!=train_pred2$device_id);sum(test_pred3$device_id!=test_pred2$device_id)
sum(train_pred4$device_id!=train_pred2$device_id);sum(test_pred4$device_id!=test_pred2$device_id)
sum(train_pred5$device_id!=train_pred2$device_id);sum(test_pred5$device_id!=test_pred2$device_id)
sum(train_pred6$device_id!=train_pred2$device_id);sum(test_pred6$device_id!=test_pred2$device_id)
sum(train_pred7$device_id!=train_pred2$device_id);sum(test_pred7$device_id!=test_pred2$device_id)
sum(train_pred8$device_id!=train_pred2$device_id);sum(test_pred8$device_id!=test_pred2$device_id)
sum(train_pred9$device_id!=train_pred2$device_id);sum(test_pred9$device_id!=test_pred2$device_id)
sum(train_pred9$device_id!=train_pred2$device_id);sum(test_pred9$device_id!=test_pred2$device_id)
sum(train_pred10$device_id!=train_pred2$device_id);sum(test_pred10$device_id!=test_pred2$device_id)
sum(train_pred11$device_id!=train_pred2$device_id);sum(test_pred11$device_id!=test_pred2$device_id)
sum(train_pred12$device_id!=train_pred2$device_id);sum(test_pred12$device_id!=test_pred2$device_id)

train_pred2$device_id=NULL; test_pred2$device_id=NULL
train_pred3$device_id=NULL; test_pred3$device_id=NULL
train_pred4$device_id=NULL; test_pred4$device_id=NULL
train_pred5$device_id=NULL; test_pred5$device_id=NULL
train_pred6$device_id=NULL; test_pred6$device_id=NULL
train_pred7$device_id=NULL; test_pred7$device_id=NULL
train_pred8$device_id=NULL; test_pred8$device_id=NULL
train_pred9$device_id=NULL; test_pred9$device_id=NULL
train_pred10$device_id=NULL; test_pred10$device_id=NULL
train_pred11$device_id=NULL; test_pred11$device_id=NULL
train_pred12$device_id=NULL; test_pred12$device_id=NULL

train_pred1=as.data.frame(train_pred1); test_pred1=as.data.frame(test_pred1)
train_pred2=as.data.frame(train_pred2); test_pred2=as.data.frame(test_pred2)
train_pred3=as.data.frame(train_pred3); test_pred3=as.data.frame(test_pred3)
train_pred4=as.data.frame(train_pred4); test_pred4=as.data.frame(test_pred4)
train_pred5=as.data.frame(train_pred5); test_pred5=as.data.frame(test_pred5)
train_pred6=as.data.frame(train_pred6); test_pred6=as.data.frame(test_pred6)
train_pred7=as.data.frame(train_pred7); test_pred7=as.data.frame(test_pred7)
train_pred8=as.data.frame(train_pred8); test_pred8=as.data.frame(test_pred8)
train_pred9=as.data.frame(train_pred9); test_pred9=as.data.frame(test_pred9)
train_pred10=as.data.frame(train_pred10); test_pred10=as.data.frame(test_pred10)
train_pred11=as.data.frame(train_pred11); test_pred11=as.data.frame(test_pred11)
train_pred12=as.data.frame(train_pred12); test_pred12=as.data.frame(test_pred12)

###########################################################
# Function
###########################################################

SQWKfun = function(x = rep(0.08, 10), t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15, truth, pred_matrix) {
  #x= init
  # x=optCuts$par
  #t1=tr1;t2=tr2;t3=tr3;t4=tr4;t5=tr5;t6=tr6;t7=tr7;t8=tr8;t9=tr9;t10=tr10;t11=tr11;t12=tr12#
  #pred_matrix=preds1
  
  for(i in 1:12){
    pred_matrix[,i]=t1[,i]*x[1] + t2[,i]*x[2] + t3[,i]*x[3] + t4[,i]*x[4] + t5[,i]*x[5] + t6[,i]*x[6] + t7[,i]*x[7] + 
      t8[,i]*x[8]  + t9[,i]*x[9] + t10[,i]*x[10]  + t11[,i]*x[11] + t12[,i]*x[12] + t13[,i]*x[13]+ t14[,i]*x[14]+ t15[,i]*x[15]
  }
  pred_matrix[pred_matrix<0]=0 
  
  #class(preds)
  err = mlogloss(truth, pred_matrix)
  cat(err, ",")
  return(err)
}


###########################################################
# Preparing environment and prediction matrices
###########################################################

n.preds <- 12
preds <- matrix(0, nrow = nrow(train_pred1), ncol = n.preds)
predsTest <- matrix(0, nrow = nrow(test_pred1), ncol = n.preds)
#colnames(predsTest)=colnames(predsTest)=1:n.preds; 

column_order=colnames(train_pred1)[2:13]
y=as.factor(train_tools$group)
w_events=train_pred3$with_events
w_events_te=test_pred3$with_events

###########################################################
# With EVents
###########################################################

tr1=train_pred1[w_events==1,2:13][, column_order]
tr2=train_pred2[w_events==1,1:12][, column_order]
tr3=train_pred3[w_events==1,1:12][, column_order]
tr4=train_pred4[w_events==1,1:12][, column_order]
tr5=train_pred5[w_events==1,1:12][, column_order]
tr6=train_pred6[w_events==1,1:12][, column_order]
tr7=train_pred6[w_events==1,13:24][, column_order]
tr8=train_pred7[w_events==1,1:12][, column_order]
tr9=train_pred8[w_events==1,1:12][, column_order]
tr10=train_pred9[w_events==1,1:12][, column_order]
tr11=train_pred10[w_events==1,1:12][, column_order]
tr12=train_pred10[w_events==1,13:24][, column_order]
tr13=train_pred11[w_events==1,1:12][, column_order]
tr14=train_pred12[w_events==1,1:12][, column_order]
tr15=train_pred12[w_events==1,13:24][, column_order]

te1=test_pred1[w_events_te==1,2:13][, column_order]
te2=test_pred2[w_events_te==1,1:12][, column_order]
te3=test_pred3[w_events_te==1,1:12][, column_order]
te4=test_pred4[w_events_te==1,1:12][, column_order]
te5=test_pred5[w_events_te==1,1:12][, column_order]
te6=test_pred6[w_events_te==1,1:12][, column_order]
te7=test_pred6[w_events_te==1,13:24][, column_order] 
te8=test_pred7[w_events_te==1,1:12][, column_order]
te9=test_pred8[w_events_te==1,1:12][, column_order]
te10=test_pred9[w_events_te==1,1:12][, column_order]
te11=test_pred10[w_events_te==1,1:12][, column_order]
te12=test_pred10[w_events_te==1,13:24][, column_order]
te13=test_pred11[w_events_te==1,1:12][, column_order]
te14=test_pred12[w_events_te==1,1:12][, column_order]
te15=test_pred12[w_events_te==1,13:24][, column_order]

preds1=preds[w_events==1,]
truth=(as.numeric(y)-1)[w_events==1]

#mlogloss(truth, tr5)

init=c(0.12982638,0.39901993,0.24086309,-0.10202335,0.04740062,-0.16810563,0.10164117,0.12458649,-0.01005444,-0.11292365,
       -0.04805931,0.63688345,0.19246062,0,0)
optCuts = optim(init, SQWKfun, t1=tr1,t2=tr2,t3=tr3,t4=tr4,t5=tr5,t6=tr6,t7=tr7,t8=tr8,t9=tr9,
                t10=tr10,t11=tr11,t12=tr12, t13=tr13,t14=tr14,t15=tr15, truth=truth, pred_matrix=preds1)
optCuts

optCuts = optim(optCuts$par, SQWKfun, t1=tr1,t2=tr2,t3=tr3,t4=tr4,t5=tr5,t6=tr6,t7=tr7,t8=tr8,t9=tr9,
                t10=tr10,t11=tr11,t12=tr12, t13=tr13,t14=tr14,t15=tr15, truth=truth, pred_matrix=preds1)
optCuts$par

optCuts = optim(optCuts$par, SQWKfun, t1=tr1,t2=tr2,t3=tr3,t4=tr4,t5=tr5,t6=tr6,t7=tr7,t8=tr8,t9=tr9,
                t10=tr10,t11=tr11,t12=tr12, t13=tr13,t14=tr14,t15=tr15, truth=truth, pred_matrix=preds1)
optCuts$par

#Applying weights to train:
x=optCuts$par
for(i in 1:12){
  preds[w_events==1,][,i]=tr1[,i]*x[1] + tr2[,i]*x[2] + tr3[,i]*x[3] + tr4[,i]*x[4] + tr5[,i]*x[5] + tr6[,i]*x[6] + tr7[,i]*x[7] + 
    tr8[,i]*x[8] + tr9[,i]*x[9] + tr10[,i]*x[10] + tr11[,i]*x[11] + tr12[,i]*x[12]+ tr13[,i]*x[13]+ tr14[,i]*x[14]+ tr15[,i]*x[15]
}
preds[preds<0]=0 
#sum(truth!=(as.numeric(y)-1)[w_events==1])

score=mlogloss(truth, preds[w_events==1,])
cat("With events score:", score)

score=mlogloss(as.numeric(y)-1, preds)
cat("overall Score:", score)


#Applying weights to test:
x=optCuts$par
for(i in 1:12){
  predsTest[w_events_te==1,][,i]=te1[,i]*x[1] + te2[,i]*x[2] + te3[,i]*x[3] + te4[,i]*x[4] + te5[,i]*x[5] + te6[,i]*x[6] + te7[,i]*x[7] + 
    te8[,i]*x[8] + te9[,i]*x[9] + te10[,i]*x[10] + te11[,i]*x[11] + te12[,i]*x[12]+ te13[,i]*x[13]+ te14[,i]*x[14]+ te15[,i]*x[15]
}

###########################################################
# WithOUT EVents
###########################################################

tr1=train_pred1[w_events==0,2:13][, column_order]
tr2=train_pred2[w_events==0,1:12][, column_order]
tr3=train_pred3[w_events==0,1:12][, column_order]
tr4=train_pred4[w_events==0,1:12][, column_order]
tr5=train_pred5[w_events==0,1:12][, column_order]
tr6=train_pred6[w_events==0,1:12][, column_order]
tr7=train_pred6[w_events==0,13:24][, column_order] 
tr8=train_pred7[w_events==0,1:12][, column_order]
tr9=train_pred8[w_events==0,1:12][, column_order]
tr10=train_pred9[w_events==0,1:12][, column_order]
tr11=train_pred10[w_events==0,1:12][, column_order]
tr12=train_pred10[w_events==0,13:24][, column_order]
tr13=train_pred11[w_events==0,1:12][, column_order]
tr14=train_pred12[w_events==0,1:12][, column_order]
tr15=train_pred12[w_events==0,13:24][, column_order]

te1=test_pred1[w_events_te==0,2:13][, column_order]
te2=test_pred2[w_events_te==0,1:12][, column_order]
te3=test_pred3[w_events_te==0,1:12][, column_order]
te4=test_pred4[w_events_te==0,1:12][, column_order]
te5=test_pred5[w_events_te==0,1:12][, column_order]
te6=test_pred6[w_events_te==0,1:12][, column_order]
te7=test_pred6[w_events_te==0,13:24][, column_order]
te8=test_pred7[w_events_te==0,1:12][, column_order]
te9=test_pred8[w_events_te==0,1:12][, column_order]
te10=test_pred9[w_events_te==0,1:12][, column_order]
te11=test_pred10[w_events_te==0,1:12][, column_order]
te12=test_pred10[w_events_te==0,13:24][, column_order]
te13=test_pred11[w_events_te==0,1:12][, column_order]
te14=test_pred12[w_events_te==0,1:12][, column_order]
te15=test_pred12[w_events_te==0,13:24][, column_order]

# truth=(as.numeric(y)-1)[w_events==0]
# score=mlogloss(truth, tr5)
# cat(score)

preds1=preds[w_events==0,]
truth=(as.numeric(y)-1)[w_events==0]
init=c(00.51192941,0.12134060,0.19637943,-0.65837175,0.36465691,0.04422553,0.04425683,0.12945291,0.28672951,-0.10031250,
       0.3667826,0.20252612,0.09198748,0,0)

opt_ne = optim(init, SQWKfun, t1=tr1,t2=tr2,t3=tr3,t4=tr4,t5=tr5,t6=tr6,t7=tr7,t8=tr8,t9=tr9,
               t10=tr10,t11=tr11,t12=tr12,t13=tr13,t14=tr14,t15=tr15, truth=truth, pred_matrix=preds1)
opt_ne

opt_ne = optim(opt_ne$par, SQWKfun, t1=tr1,t2=tr2,t3=tr3,t4=tr4,t5=tr5,t6=tr6,t7=tr7,t8=tr8,t9=tr9,
               t10=tr10,t11=tr11,t12=tr12,t13=tr13,t14=tr14,t15=tr15, truth=truth, pred_matrix=preds1)
opt_ne

opt_ne = optim(opt_ne$par, SQWKfun, t1=tr1,t2=tr2,t3=tr3,t4=tr4,t5=tr5,t6=tr6,t7=tr7,t8=tr8,t9=tr9,
               t10=tr10,t11=tr11,t12=tr12,t13=tr13,t14=tr14,t15=tr15, truth=truth, pred_matrix=preds1)
opt_ne

#Applying weights to train:
x=opt_ne$par
for(i in 1:12){
  preds[w_events==0,][,i]=tr1[,i]*x[1] + tr2[,i]*x[2] + tr3[,i]*x[3] + tr4[,i]*x[4] + tr5[,i]*x[5] + tr6[,i]*x[6] + tr7[,i]*x[7] + 
    tr8[,i]*x[8] + tr9[,i]*x[9] + tr10[,i]*x[10] + tr11[,i]*x[11] + tr12[,i]*x[12]+ tr13[,i]*x[13]+ tr14[,i]*x[14]+ tr15[,i]*x[15]
}
preds[preds<0]=0
#sum(truth!=(as.numeric(y)-1)[w_events==1])

score=mlogloss(truth, preds[w_events==0,])
cat("WithOUT events score:", score)

score=mlogloss(as.numeric(y)-1, preds)
cat("overall Score:", score)
# overall Score: 2.241608
#2.23751 first try, 8 models
#2.23751
#2.237351

##Predicting test set - No events
x=opt_ne$par
for(i in 1:12){
  predsTest[w_events_te==0,][,i]=te1[,i]*x[1] + te2[,i]*x[2] + te3[,i]*x[3] + te4[,i]*x[4] + te5[,i]*x[5] + te6[,i]*x[6] + te7[,i]*x[7] + 
    te8[,i]*x[8] + te9[,i]*x[9] + te10[,i]*x[10] + te11[,i]*x[11] + te12[,i]*x[12]+ te13[,i]*x[13]+ te14[,i]*x[14]+ te15[,i]*x[15]
}
predsTest[predsTest<0]=0 



###########################################################
# Submission
###########################################################
submit<-cbind(device_id=test_pred1$device_id, as.data.frame(predsTest))
scaled_preds=submit[,2:13]/rowSums(submit[,2:13])
submit2=data.frame(device_id=submit$device_id, scaled_preds)

colnames(submit2)[2:13]=levels(y)
write.csv(submit2, '../no_leak_sub/best_submission_no_leak.csv', quote=F, row.names=F) 

##

optCuts
opt_ne

