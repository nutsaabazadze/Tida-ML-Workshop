remove(list = ls())
### Load packages
library(xgboost)
library(useful)
library(dygraphs)

### Load the data (So that this file is self contained)
land_train <- readRDS('data/manhattan_Train.rds')
land_test <- readRDS('data/manhattan_Test.rds')
land_val <- readRDS('data/manhattan_Validate.rds')

### Set seed for reproducibility
set.seed(1123)

### Have a look at variable historic district
table(land_train$HistoricDistrict)

histFormula <- HistoricDistrict ~ FireService + 
  ZoneDist1 + ZoneDist2 + Class + LandUse + 
  OwnerType + LotArea + BldgArea + ComArea + 
  ResArea + OfficeArea + RetailArea + 
  GarageArea + FactryArea + NumBldgs + 
  NumFloors + UnitsRes + UnitsTotal + 
  LotFront + LotDepth + BldgFront + 
  BldgDepth + LotType + Landmark + BuiltFAR +
  Built + TotalValue - 1

??xgb.train

landX_train <- build.x(histFormula, data=land_train,
                       contrasts=FALSE, sparse=TRUE)


landY_train <- build.y(histFormula, data=land_train) %>% 
  as.integer() - 1
head(landY_train, n=15)


landX_test <- build.x(histFormula, data=land_test,
                      contrasts=FALSE, sparse=TRUE)
landY_test <- build.y(histFormula, data=land_test) %>% 
  as.integer() - 1

landX_val <- build.x(histFormula, data=land_val,
                     contrasts=FALSE, sparse=TRUE)
landY_val <- build.y(histFormula, data=land_val) %>% 
  as.integer() - 1

xgTrain <- xgb.DMatrix(data=landX_train, label=landY_train)
xgVal <- xgb.DMatrix(data=landX_val, label=landY_val)

#xgb.train is an advanced interface for training an 
#xgboost model. The xgboost function is a simpler 
#wrapper for xgb.train.

xg1 <- xgb.train(
  data=xgTrain,
  #logistic regression for binary classification. Output probability
  objective='binary:logistic',
  #max number of boosting iterations
  nrounds=1
)



xg2 <- xgb.train(
  data=xgTrain,
  objective='binary:logistic',
  nrounds=1,
  # negative log-likelihood
  eval_metric='logloss',
  watchlist=list(train=xgTrain)
)

### Log Loss 0.58

xg3 <- xgb.train(
  data=xgTrain,
  objective='binary:logistic',
  nrounds=100,
  eval_metric='logloss',
  watchlist=list(train=xgTrain)
)


xg4 <- xgb.train(
  data=xgTrain,
  objective='binary:logistic',
  nrounds=300,
  eval_metric='logloss',
  watchlist=list(train=xgTrain)
)


xg5 <- xgb.train(
  data=xgTrain,
  objective='binary:logistic',
  nrounds=300,
  eval_metric='logloss',
  watchlist=list(train=xgTrain, validate=xgVal)
)


# early_stopping_rounds - If set to an integer k, 
# training with a validation set will stop if the performance doesn't 
# improve for k rounds.

xg6 <- xgb.train(
  data=xgTrain,
  objective='binary:logistic',
  nrounds=300,
  eval_metric='logloss',
  watchlist=list(train=xgTrain, validate=xgVal),
  early_stopping_rounds=70
)


### Let us Play with another parameter ###

#max_depth maximum depth of a tree. Default: 6
xg7 <- xgb.train(
  data=xgTrain,
  objective='binary:logistic',
  nrounds=1000,
  eval_metric='logloss',
  watchlist=list(train=xgTrain, validate=xgVal),
  early_stopping_rounds=70,
  max_depth=8
)

### Needed only 108 iteration to achieve the minimum log loss of 0.3347

xg8 <- xgb.train(
  data=xgTrain,
  objective='binary:logistic',
  nrounds=1000,
  eval_metric='logloss',
  watchlist=list(train=xgTrain, validate=xgVal),
  early_stopping_rounds=70,
  max_depth=3
)

### needed 493 iterations to achieve the minimum log loss of 0.3518


xg9 <- xgb.train(
  data=xgTrain,
  objective='binary:logistic',
  nrounds=50,
  eval_metric='logloss',
  watchlist=list(train=xgTrain, validate=xgVal),
  early_stopping_rounds=70,
  max_depth=3,
  #Setting it to 0.5 means that xgboost randomly collected half of the 
  #data instances to grow trees and this will prevent overfitting. 
  #It makes computation shorter (because less data to analyse). 
  subsample=0.5, 
  # for each tree randomly choose only half of the columns
  colsample_bytree=0.5,
  # number of trees to grow per round. 
  # Useful to test Random Forest through Xgboost
  # 50 trees at a time 
  num_parallel_tree=50
)

# It takes a lot of time better to do just boosted trees