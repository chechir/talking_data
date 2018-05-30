
import numpy as np

## set seed for reproducability
seed = 123
np.random.seed(seed)

import os
import pandas as pd
from scipy import sparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

## Batch generators ##################################################################################################################################

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

## path of working directory
PATH = '../danijel'

app_events = pd.read_csv(PATH + '/input/app_events.csv', dtype = {'device_id': np.str})

app_events = app_events.groupby('event_id')['app_id'].apply(
    lambda x: ' '.join(set('app_id:' + str(s) for s in x)))

events = pd.read_csv(PATH + '/input/events.csv', dtype = {'device_id': np.str})
events['app_id'] = events['event_id'].map(app_events)
events = events.dropna()

del(app_events)

events = events[['device_id', 'app_id']]
events.loc[:,'device_id'].value_counts(ascending = True)

events = events.groupby('device_id')['app_id'].apply(
    lambda x: ' '.join(set(str(' '.join(str(s) for s in x)).split(' '))))
events = events.reset_index(name='app_id')

events = pd.concat([pd.Series(row['device_id'], row['app_id'].split(' '))
                    for _, row in events.iterrows()]).reset_index()
events.columns = ['app_id', 'device_id']

f3 = events[['device_id', 'app_id']]  

## app labels

app_labels = pd.read_csv(PATH + '/input/app_labels.csv')
label_cat = pd.read_csv(PATH + '/input/label_categories.csv')
label_cat = label_cat[['label_id','category']]

app_labels = app_labels.merge(label_cat, on = 'label_id', how = 'left')
app_labels = app_labels.groupby(['app_id','category']).agg('size').reset_index()
app_labels = app_labels[['app_id','category']]

events['app_id'] = events['app_id'].map(lambda x : x.lstrip('app_id:'))
events['app_id'] = events['app_id'].astype(str)
app_labels['app_id'] = app_labels['app_id'].astype(str)

events = pd.merge(events, app_labels, on = 'app_id', how = 'left').astype(str)

events = events.groupby(['device_id','category']).agg('size').reset_index()
events = events[['device_id','category']]

f5 = events[['device_id', 'category']] 

## phone brand
pbd = pd.read_csv(PATH + '/input/phone_brand_device_model.csv', dtype = {'device_id': np.str})
pbd.drop_duplicates('device_id', keep = 'first', inplace = True)

## train and test set

train = pd.read_csv(PATH + '/input/gender_age_train.csv', dtype = {'device_id': np.str})
train.drop(['age', 'gender'], axis = 1, inplace = True)
train = train.sort_values('device_id')

test = pd.read_csv(PATH + '/input/gender_age_test.csv', dtype = {'device_id': np.str})
test['group'] = np.nan
test = test.sort_values('device_id')

split_len = len(train)

## group labels
ytrain = train['group']
label_group = LabelEncoder()
ytrain = label_group.fit_transform(ytrain)
device_id = test['device_id']

# concat train and test
df = pd.concat((train, test), axis = 0, ignore_index = True)
df = pd.merge(df, pbd, how = 'left', on = 'device_id')

df['phone_brand'] = df['phone_brand'].apply(lambda x: 'phone_brand:' + str(x))
df['device_model'] = df['device_model'].apply(lambda x: 'device_model:' + str(x))

## add features

f1 = df[['device_id', 'phone_brand']]   
f2 = df[['device_id', 'device_model']]  

del(events, df)

f1.columns.values[1] = 'feature'
f2.columns.values[1] = 'feature'
f5.columns.values[1] = 'feature'
f3.columns.values[1] = 'feature'

FLS = pd.concat((f1, f2, f3, f5), axis = 0, ignore_index = True)

## create sparse matrix (1HE)

device_ids = FLS['device_id'].unique()
features = FLS['feature'].unique()

ones = np.ones(len(FLS))

dec = LabelEncoder().fit(FLS['device_id'])
row = dec.transform(FLS['device_id'])
col = LabelEncoder().fit_transform(FLS['feature'])

sparse_matrix = sparse.csr_matrix((ones, (row, col)), shape = (len(device_ids), len(features)))
sparse_matrix = sparse_matrix[:, sparse_matrix.getnnz(0) > 0]

del(FLS, ones, f1, f2, f3, f5, events)

## sparse train/test data

train_row = dec.transform(train['device_id'])
train_sp = sparse_matrix[train_row, :]

test_row = dec.transform(test['device_id'])
test_sp = sparse_matrix[test_row, :]

device_id_train = train['device_id'].values
device_id_test = test['device_id'].values

## read folds
folds = pd.read_csv(PATH + '/folds/folds_10.csv')['fold'].values

## neural net
def nn_model():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=train_sp.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, input_dim=train_sp.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init='normal', activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return(model)

## cv params
nfolds = np.max(folds)
nbags = 1

p_group = np.zeros((train_sp.shape[0], 12))

for i in range(1, nfolds+1):
    ## cv index
    inTr = [idx for idx, fold in enumerate(folds) if fold != i]
    inTe = [idx for idx, fold in enumerate(folds) if fold == i]
    ## train data
    xtr = train_sp[inTr]
    ytr = ytrain[inTr]
    ## validation data
    xval = train_sp[inTe]
    yval = ytrain[inTe]
    ## object to store predictions
    pred = np.zeros((xval.shape[0], 12))
    for j in range(nbags):
        model = nn_model()
        ## training
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 400, True),
                                  nb_epoch = 18,
                                  samples_per_epoch = 69984,
                                  verbose = 0)
        ## prediction
        pred += model.predict_generator(generator = batch_generatorp(xval, 800, False), val_samples = xval.shape[0])
    ## average predictions
    pred /= nbags
    p_group[inTe] = pred
    score = log_loss(yval, pred)
    print('Fold ', i, '-', score, '\n')

score = log_loss(Y, p_group)
print('Total score', score)

## save cv predictions
col_labels = ('F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+', 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+')       
df = pd.DataFrame(p_group, columns = col_labels)
device_id_train = pd.read_csv(PATH + '/input/for_py_train_device_id_all.csv')['x'].values
df['device_id'] = device_id_train
df = df.set_index('device_id')
df.to_csv(PATH + '/keras_3_cv10_train.csv', index = True, index_label = 'device_id')

