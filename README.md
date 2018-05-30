

### <center> TalkingData Competition - 3rd Place Solution of Team utc(+1,-3) </center>
### <center> By Danijel Kivaranovic and Matias Thayer </center>

#### 1. Overview

We used Python and R to build our solution. Particularly, for Python we used `Keras`  for neural networks and `sklearn` for logistic regression. In R we mainly used `XGBoost` for Gradient boosting algorithms and the base function *optim* was used for model ensembling. In the Chapters 2 and 3 model descriptions and instructions on how to build them are given. In Chapter 4 we show the ensembling approach of our best *no leak* submission. Chapter 5 gives the details on the best *leak* submission. Only the models in Chapter 5 use information about the leak.

#### 2. Danijel Kivaranovic

Duplicate rows were removed and all files were saved as .RData files to reload faster. The different approaches for xgboost and keras are explained below. Out of fold predictions and test predictions were made for each model.

##### Xgboost

Two different models were trained for devices with and without events. To make the predictions for devices without events we trained on all devices. The only features were phone brand, device model and event flag (has event: yes/no).

We only trained on the subset of devices with events to predict IDs with events. We can subdivide the features in three groups: (1) features related to categories, (2) to apps and to (3) event information. For each device

1. count how often each category appears in the app list (all apps that can be assigned to the device)
2. count how often each app appears in the event list
3. calculate median latitude and longitude of events. Count at which hour and at which weekday the events happened.

Only categories with >25 non-zero entries and the 1000 most frequent apps were used for training. Those thresholds achieved the best cv score. Further, phone brand and device model were included. In total, we have 1423 columns.


The same kind of 2-stage model was fitted for devices with/without events, just with different parameters.

1. Predict the probability of gender
2. Use gender as additional feature and predict the probability of age groups

The clue is that we do not need the gender in the test set to make the prediction, because we have to make two predictions for each ID in the test set anyway. First, we assume that the user is female and predict, then we assume that the user is male and predict again. Using the definition of conditional probability, we combine the predictions for gender and age groups to get the probability for each group:

$P(A_i, F) = P(A_i| F) P(F)$ for $i = 1,\dots,6$ and
$P(A_i, M) = P(A_i| M) P(M)$ for $i = 1,\dots,6$,

where $A_i$ denote the age groups 1 to 6, and $F$ and $M$ denote female and male, respectively.

##### Keras

The models keras\_1 and keras\_3 were trained on the whole training set, just with different parameters. For the keras\_2 model, we used the same splitting schema as for the xgboost model and trained a 2-stage model for devices without events. However, splitting between devices with/without events did not improve the score.

All features were one-hot encoded. The features were event flag (has event: yes/no), category features (has category: yes/no), app features (has app: yes/no), phone brand and device model. No feature selection was performed. In total, the training set had 15772 columns.


##### Build the models
Make sure that all training files are saved in **danijel/input**.
Open a R console and set the working directory to **danijel**. At first, run "create\_rdata.R" and afterwards "for\_py\_create\_train\_test.R".
To build the xgb model, run all scripts in the folder **danijel/xgb**. Of importance, the script "xgb\_merge.R" has to be run at last.
To build all python models, open a command line, change directory to **danijel** and run all scripts in the folders **danijel/keras\_1**, **danijel/keras\_2** and **danijel/keras\_3**. Of importance, the R script "keras\_2\_merge.R" has again to be run after all other scripts in **danijel/keras\_2**. 
Train and test predictions were all saved in **danijel/preds**.

#### 3. Matias Thayer

#### Keras model and logistic regression for sparse matrix
Our best "single" model was a combination of a simple logistic regression for devices without events and a neural network model for devices with events. It was trained using one hot encoding.

###### Features

1. Dummies for brands, models, and app_id
2. TF-IDF of brand and model (for devices without events)
3. TF-IDF of brand, model and labels (for devices with events)
4. Frequency of brands and model names (that one produced a small but clear improvement)
5. Dummies for hour of the day
6. Dummies for count of events
7. Simple binary flag to indicate if the device has events or not

After removing features with zero variance the number of features was 18.150 for training devices with events and 3.408 for training devices without events

###### Fitting the model

This model was fit using 10 folds. Each fold model predicted on the test set and the predictions were averaged to get the final prediction.

The model parameters for this keras model were:

    model = Sequential()
    model.add(Dropout(0.4, input_shape=(num_columns,)))
    model.add(Dense(75))
    model.add(PReLU())
    model.add(Dropout(0.30))
    model.add(Dense(50, init='normal', activation='tanh'))
    model.add(PReLU())
    model.add(Dropout(0.20))
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

The number of epochs was 20 and the samples per epoch was the size of the train set (for the corresponding fold)

###### Score on the Leaderboard
This Keras/Logistic regression model achieved a score of 2.23452 on public Leaderboard and 2.23822 on private Leaderboard. On our final blending a new version of this model was included but for folds 5, and also the predictions were averaged by running the same script several times with different seeds


#### Other models
In our best submission blend we also noticed a small improvement by adding some additional models with lower performances. Those "second order" models included:

1. Additional keras with different configuration (+1 model)
2. Xgboosts using a different set of features (+3 models)
3. Elastic net model using glmnet and the different set of features (+1 model)

##### Build the models
Make sure that all training files are saved in **matias/input**, including the fold_10.csv and fold_5.csv files (already provided)
For R files, open a R console and set the working directory to **matias**. For Python files, open Idle or your favorite IDE (I used Pycharm community) and set the project directory to **matias**. Then run the scripts in the following order:

######Preprocessing data
1. 1-BM_Low RAM bag-of-apps_data.py *# Generates some PCA of bag features*
2. 2-lvl0_data_preparation.R *# Calculate some ratio of app labels for each device*

######Generating model predictions in R
3. cv5_shityGlmnet_1hot_d1-d31.R *# Train an elastic net model*
4. cv5_shityXgb_1hot_d1-d31_bagged.R *# Train an xgboost model*
5. cv5_shityXgb_1hot_d1-d31_bagged_seed2.R *# Train the same xgboost model with different seed*
6. cv5_shityXgb_2_1hot_d1-d31_bagged.R *# Train an xgboost model*
7. cv5_shityXgbLinear_1hot_d1-d31_bagged.R *# Train an xgboost model using gblinear*

######Generating model predictions in Python
8. keras_cv5_2_bagging_split.py *# Train a keras model*
9. keras_cv10_with_bags5.py *# Train a keras model*
10. keras_cv10_with_bags5_wEvents_AllData2.py *# Train a keras model*
11. keras_cv10_with_bags5_wEvents_AllData2_seed1.py *# Train the same keras with different seed*
12. keras_cv10_with_bags5_wEvents_AllData2_seed2.py *# Train the same keras with different seed*

Train and test predictions were all saved in **matias/preds**.

#### 4. Best *no leak* submission
######Blending predictions using the optim package in R
At this point you should have all the prediction files in **danijel/preds** and **matias/preds**. Then the following script needs to run:

- matias/blendcv15_combine_optim_x15.R

The final blending will be generated in the folder: **no_leak_sub**. This folder lies at the same level than **matias** and **danijel** folders. This script will optimize the weights for the previous 15 predictions generated. The script performs two independent optimizations: One for for devices with events and another one for devices with events.


#### 5. Best *leak* submission

We observed that the row leak only improved the score of devices without events. For devices with events, we got slightly worse scores. Therefore, the predictions of the *leak* solution are the same as the *no leak* solution for devices with events. Additionally, Keras performed worse than Xgboost with the leak. So we only trained the same Xgboost model for no event devices, just with the normalized row as an additional feature included. This gave us 2.16835	on public LB and 2.17584 on private LB.
The final score of 2.14223 on public LB and	2.14949 on private LB was achieved with a matching script where we tried to exactly match IDs in the train set with IDs in the test set using the normalized rows.

##### Build the models
Open a R console and set the working directory to the root folder. The scripts have to be run in the following order: 1) 'create\_rdata\_leak.R' 2) 'xgb\_test\_leak.R' 3) 'match\_ids\_leak.R'. The final submission will be saved in the folder **leak\_sub**.


#### Apendix

#####Hardware and Software

######Danijel's R scripts were run using the following environment:

- Operating System: Windows 8.1, 64 bits
- CPU: Intel(R) Core(TM) i5-4460 CPU @ 3.20GHz
- RAM: 6 GB
- R version 3.2.4 Revised (2016-03-16 r70336)
- Used Packages: *Matrix_1.2-4, xgboost_0.4-3, lubridate_1.5.6, ggplot2_2.1.0, stringr_1.0.0, tidyr_0.4.1, dplyr_0.4.3, data.table_1.9.6*

######Danijel's Python scripts were run using the following environment:

- Operating System: Windows 8.1, 64 bits
- CPU: Intel(R) Core(TM) i5-4460 CPU @ 3.20GHz
- RAM: 6 GB
- Python 3.4.5 :: Anaconda custom (64-bit)
- Used Packages: *Keras 1.0.7, pandas 0.18.1, scipy 0.18.0, scikit-learn 0.17.1*

######Matias' R scripts were run using the following environment:

- Operating System: Windows Server 2012 R2 Staandard Edition, 64 bits
- CPU: Intel(R) Xeon(R) CPU ES-1620 v2 @ 3.79GHz
- RAM: 64 GB
- R version 3.3.1 (2016-06-21) "Bug in Your Hair"
- Used Packages: *xgboost_0.4-4, bit_1.1-12, stringr_1.0.0, dplyr_0.5.0, data.table_1.9.6, Matrix_1.2-6, glmnet_2.0-5, caret_6.0-70*


######Matias' Python scripts were run using the following environment:

- Operating System: Ubuntu 14.04 LTS, 64 bits
- CPU: Intel(R) Core(TM) i5-4200U CPU @ 1.60GHz x4
- RAM: 16 GB
- Python 2.7.12 :: Anaconda custom (64-bit)
- Used Packages: *Keras 1.0.2 (Theano backend), pandas 0.18.1, scipy 0.17.1, scikit-learn 0.17.1*

