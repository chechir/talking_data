
## load libraries
import numpy as np

## set seed for reproducability
seed = 7
np.random.seed(seed)

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from scipy import sparse
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

### Features were generated in R. 

## path of working directory
PATH = '../danijel'

## read folds
folds = pd.read_csv(PATH + '/folds/folds_5.csv')['fold'].values

## read train features
triples_train = pd.read_csv(PATH + '/input/for_py_xtrain_triples_all.csv')

nrow = np.max(triples_train['row']) + 1
ncol = np.max(triples_train['col']) + 1

xtrain = sparse.csr_matrix(
    (triples_train['x'], (triples_train['row'], triples_train['col'])), shape = (nrow, ncol))

## subset
first_cols = 1559
xtrain = xtrain[:,range(first_cols)]

## add gender to xtrain
gender = pd.read_csv(PATH + '/input/for_py_gender_all.csv')['x'].values
triples = pd.read_csv(PATH + '/input/for_py_xtrain_triples_all.csv')
triples['col'] += 1
row = [idx for idx, sex in enumerate(gender) if sex == 1]
col = np.repeat(0, len(row))
value = np.repeat(1, len(row))
add_df = pd.DataFrame({'row': row, 'col': col, 'x': value})

triples = triples.append(add_df)
nrow = np.max(triples['row']) + 1
ncol = np.max(triples['col']) + 1
xtrain_wg = sparse.csr_matrix(
    (triples['x'], (triples['row'], triples['col'])), shape = (nrow, ncol))[:,range(first_cols + 1)]

## modified validation data
xtrain_mod = xtrain_wg
xtrain_mod[:,0] = np.transpose(sparse.csr_matrix(np.repeat(0, xtrain_mod.shape[0])))
xtrain_mod1 = xtrain_mod
xtrain_mod[:,0] = np.transpose(sparse.csr_matrix(np.repeat(1, xtrain_mod.shape[0])))
xtrain_mod2 = xtrain_mod
del(xtrain_mod)
    
## clear workspace
del(triples_train)

## neural nets
def nn_model1():
    model = Sequential()
    model.add(Dense(100, input_dim = xtr.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(2, init = 'he_normal'))
    model.add(Activation('softmax'))
    adagrad = Adagrad(lr = 0.005, epsilon = 1e-08)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = adagrad, metrics = ['accuracy'])
    return(model)

def nn_model2():
    model = Sequential()
    model.add(Dense(100, input_dim = xtr.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(6, init = 'he_normal'))
    model.add(Activation('softmax'))
    adagrad = Adagrad(lr = 0.005, epsilon = 1e-08)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = adagrad, metrics = ['accuracy'])
    return(model)

## cv params
nfolds = np.max(folds)
nbags = 5

## objects to store preds
p_gender = np.zeros((xtrain.shape[0], 2))
p_age_group = np.zeros((xtrain.shape[0], 12))

for i in range(1, nfolds+1):
    #--------------------------------------------------------------------------------------------------------------
    # neural net - gender 
    #--------------------------------------------------------------------------------------------------------------
    ## cv index
    inTr = [idx for idx, fold in enumerate(folds) if fold != i]
    inTe = [idx for idx, fold in enumerate(folds) if fold == i]
    ## gender
    y = pd.read_csv(PATH + '/input/for_py_gender_all.csv')['x'].values
    ## train data
    xtr = xtrain[inTr]
    ytr = y[inTr]
    ## validation data
    xval = xtrain[inTe]
    yval = y[inTe]
    ## pred object
    pred = np.zeros((len(inTe), 2))
    for j in range(nbags):
        model = nn_model1()
        ## training
        model.fit_generator(generator = batch_generator(xtr, ytr, 400, True),
                            nb_epoch = 15,
                            samples_per_epoch = 20000,
                            verbose = 0)
        ## prediction
        pred += model.predict_generator(generator = batch_generatorp(xval, 800, False), val_samples = xval.shape[0])
    ## average predictions
    pred /= nbags
    p_gender[inTe] = pred
    score = log_loss(yval, pred)
    print('Iter ', i, '- Gender -', score, '\n')
    #--------------------------------------------------------------------------------------------------------------
    # neural net - age group
    #--------------------------------------------------------------------------------------------------------------
    y = pd.read_csv(PATH + '/input/for_py_agegroup_all.csv')['x'].values
    ## train data
    xtr = xtrain_wg[inTr]
    ytr = y[inTr]
    ## validation data
    xval = xtrain_wg[inTe]
    yval = y[inTe]
    ## modified validation data
    xval_mod1 = xtrain_mod1[inTe]
    xval_mod2 = xtrain_mod2[inTe]
    ## pred objects
    p1 = np.zeros((len(inTe), 6))
    p2 = np.zeros((len(inTe), 6))
    p_tmp = np.zeros((len(inTe), 6))
    for j in range(nbags):
        model = nn_model2()
        ## training
        model.fit_generator(generator = batch_generator(xtr, ytr, 400, True),
                            nb_epoch = 30,
                            samples_per_epoch = 40000,
                            #validation_data = (xval.todense(), yval), 
                            verbose = 0)
        ## prediction
        p_tmp += model.predict_generator(generator = batch_generatorp(xval, 800, False), val_samples = xval.shape[0])
        p1 += model.predict_generator(generator = batch_generatorp(xval_mod1, 800, False), val_samples = xval_mod1.shape[0])
        p2 += model.predict_generator(generator = batch_generatorp(xval_mod2, 800, False), val_samples = xval_mod2.shape[0])
    ## average predictions
    p_tmp /= nbags
    p1 /= nbags
    p2 /= nbags
    score = log_loss(yval, p_tmp)
    print('Iter ', i, '- Age group -', score, '\n')
    ## stack age group predictions
    pred = np.column_stack([p1, p2])
    p_age_group[inTe] = pred


#--------------------------------------------------------------------------------------------------------------
# combine predictions
#--------------------------------------------------------------------------------------------------------------

s1 = p_age_group[:, range(0,6)] * p_gender[:, 0, np.newaxis]  
s2 = p_age_group[:, range(6,12)] * p_gender[:, 1, np.newaxis]  
p_group = np.column_stack([s1, s2])

y = pd.read_csv(PATH + '/input/for_py_group_all.csv')['x'].values
log_loss(y, p_group)
# 2.389798484454396

## save cv predictions
col_labels = ('F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+', 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+')       
df = pd.DataFrame(p_group, columns = col_labels)
device_id_train = pd.read_csv(PATH + '/input/for_py_train_device_id_all.csv')['x'].values
df['device_id'] = device_id_train
df = df.set_index('device_id')
df.to_csv(PATH + '/preds/keras_2_cv5_train_noevents.csv', index = True, index_label = 'device_id')


