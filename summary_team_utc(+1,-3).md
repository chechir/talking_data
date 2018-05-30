### <center> TalkingData Competition - 3rd Place Solution of Team utc(+1,-3) </center>
### <center> By Danijel Kivaranovic and Matias Thayer </center>

We used Python and R to build our solution. Particularly, for Python we used `Keras`  for neural networks and `sklearn` for logistic regression. In R we mainly used `XGBoost` for Gradient boosting algorithms and the base function *optim* was used to combine the predictions for the best submission.

Leaderboard scores of the best models are shown in the table below. 

|Model                       | Public LB    | Private LB |
|----------------------------|--------------|------------|
|Xgboost                     |2.25037       |2.25484     |
|Keras/Logistic regression   |2.23452       |2.23822     |
|Best submission (no leak)   |2.22700	    |2.23187     |
|Best sub. with xgb leak     |2.16835	    |2.17584     |
|Best submission (with leak) |2.14223	    |2.14949     |

## No leak solution

We first describe the models we used to achieve the best score without information of the leak.

### XGBoost model

Separate models were trained for devices with and without events. We first describe the approach for devices with events and then go on to the devices without events.

#### Devices with events

The model was only fitted on devices with events.

##### Features
We can subdivide the features in three groups: (1) features related to categories, (2) to apps and to (3) event information. For each device

1. count how often each category appears in the app list (all apps that can be assigned to the device).
2. count how often each app appears in the event list.
3. calculate median latitude and longitude of events. Count at which hour and at which weekday the events happened.

Only categories with >25 non-zero entries and the 1000 most frequent apps were used for training. Those thresholds achieved the best cv score. Further, phone brand and device model were included. In total, we have 1423 columns.

##### Fitting the model

A 2-stage model was build:

1. Predict the probability of gender
2. Use gender as additional feature and predict the probability of age groups

The clue is that we do not need the gender in the test set to make the prediction, because we have to make two predictions for each ID in the test set anyway. First, we assume that the user is female and predict, then we assume that the user is male and predict again. Using the definition of conditional probability, we combine the predictions for gender and age groups to get the probability for each group:

P(A_i, F) = P(A_i| F) x P(F) for i = 1,...,6 and
P(A_i, M) = P(A_i| M) x P(M) for i = 1,...,6,

where A_i are the age groups 1 to 6, and F and M are female and male, respectively.

#### Devices without events

The model was fitted on all devices but predictions were only used for devices without events.

##### Features
The model has only 3 features: phone brand, device model and event flag (has event: yes/no)

##### Fitting the model
The same 2-stage model was built as before.

#### Score on the Leaderboard
This Xgboost model achieved a score of 2.25037 on public Leaderboard and 2.25484 on private Leaderboard.

### Keras model and logistic regression for sparse matrix
Our best "single" model was a combination of a simple logistic regression for devices without events and a neural network model for devices with events. It was trained using one hot encoding.

##### Features

1. Dummies for brands, models, and app_id
2. TF-IDF of brand and model (for devices without events)
3. TF-IDF of brand, model and labels (for devices with events)
4. Frequency of brands and model names (that one produced a small but clear improvement)
5. Dummies for hour of the day
6. Dummies for count of events
7. Simple binary flag to indicate if the device has events or not

After removing features with zero variance the number of features was 18.150 for training devices with events and 3.408 for training devices without events

#### Fitting the model

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

#### Score on the Leaderboard
This Keras/Logistic regression model achieved a score of 2.23452 on public Leaderboard and 2.23822 on private Leaderboard.


### Other models
In our best submission blend we also noticed a small improvement by adding some additional models with lower performances. Those "second order" models included:

1. Additional keras with different configuration (4 models)
2. Xgboosts using a different set of features (3 models)
3. Elastic net model using glmnet and the different set of features (1 model)

## Leak Solution
We observed that the row leak only improved the score of devices without events. For devices with events, we got slightly worse scores. Therefore, the predictions of the *leak* solution are the same as the *no leak* solution for devices with events. Additionally, Keras performed worse than Xgboost with the leak. So we only trained the same Xgboost model for no event devices, just with the normalized row as an additional feature included. This gave us 2.16835	on public LB and 2.17584 on private LB.
The final score of 2.14223 on public LB and	2.14949 on private LB was achieved with a matching script where we tried to exactly match IDs in the train set with IDs in the test set using the normalized rows.


