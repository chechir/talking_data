
rm(list=ls());gc()
library(caret); 

train <- fread("./input/gender_age_train.csv",
                     colClasses=c("character","character",
                                  "integer","character"))

train=train[order(train$device_id),]

# Using same steps to keep reproducibility
y <- train$group
(group_name <- na.omit(unique(y)))
train_label <- match(y,group_name)-1
set.seed(19)


### n Folds
n.fold=10
folds<-createFolds(y=as.factor(train_label), k=n.fold, list=TRUE, returnTrain = FALSE)
folds_df=data.frame(device_id=train$device_id, fold=rep(0, nrow(train)))


for (fold.id in 1:n.fold) {
  cat("Starting fold ", fold.id, "\n")
  train.id=unlist(folds[fold.id])
  folds_df$fold[train.id]=fold.id

}

write.csv(folds_df,"./input/folds_10.csv", row.names =FALSE)
table(folds_df$fold)
