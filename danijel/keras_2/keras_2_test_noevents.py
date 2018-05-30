
## load libraries
import numpy as np

## set seed for reproducability
seed = 111
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

device_id = pd.read_csv(PATH + '/input/for_py_test_device_id_all.csv')['x'].values
triples_tr = pd.read_csv(PATH + './input/for_py_xtrain_triples_all.csv')
triples_te = pd.read_csv(PATH + './input/for_py_xtest_triples_all.csv')

## xtrain
nrow = np.max(triples_tr['row']) + 1
ncol = np.max(triples_tr['col']) + 1

xtrain = sparse.csr_matrix((triples_tr['x'], (triples_tr['row'], triples_tr['col'])), shape = (nrow, ncol))

## xtest
nrow = np.max(triples_te['row']) + 1
ncol = np.max(triples_te['col']) + 1

xtest = sparse.csr_matrix((triples_te['x'], (triples_te['row'], triples_te['col'])), shape = (nrow, ncol))

del(triples_tr, triples_te)

first_cols = 1559
xtrain = xtrain[:,range(first_cols)]
xtest = xtest[:,range(first_cols)]

#-------------------------------------------------------------------------------------------------
# training - gender
#-------------------------------------------------------------------------------------------------

y = pd.read_csv(PATH + '/input/for_py_gender_all.csv')['x'].values

## neural net
def nn_model():
    model = Sequential()
    model.add(Dense(100, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(2, init = 'he_normal'))
    model.add(Activation('softmax'))
    adagrad = Adagrad(lr = 0.005, epsilon = 1e-08)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = adagrad, metrics = ['accuracy'])
    return model


nbags = 10
p_gender = np.zeros((xtest.shape[0], 2))

for i in range(nbags):
    model = nn_model()
    fit = model.fit_generator(generator = batch_generator(xtrain, y, 400, True),
                              nb_epoch = 15,
                              samples_per_epoch = 20000,
                              verbose = 0)
    p_gender += model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])

p_gender /= nbags

#-------------------------------------------------------------------------------------------------
# training - age group
#-------------------------------------------------------------------------------------------------

gender = pd.read_csv(PATH + '/input/for_py_gender_all.csv')['x'].values
y = pd.read_csv(PATH + '/input/for_py_agegroup_all.csv')['x'].values

## add gender to train data
xtrain = sparse.csr_matrix(np.column_stack([gender, xtrain.todense()]))

## modified validation data
g0 = np.repeat(0, xtest.shape[0])
xtest_mod0 = sparse.csr_matrix(np.column_stack([g0, xtest.todense()]))

g1 = np.repeat(1, xtest.shape[0])
xtest_mod1 = sparse.csr_matrix(np.column_stack([g1, xtest.todense()]))

## clear workspace
del(xtest)

## neural net
def nn_model():
    model = Sequential()
    model.add(Dense(100, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(6, init = 'he_normal'))
    model.add(Activation('softmax'))
    adagrad = Adagrad(lr = 0.005, epsilon = 1e-08)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = adagrad, metrics = ['accuracy'])
    return model

nbags = 10
p0 = np.zeros((xtest_mod0.shape[0], 6))
p1 = np.zeros((xtest_mod1.shape[0], 6))

for i in range(nbags):
    model = nn_model()
    fit = model.fit_generator(generator = batch_generator(xtrain, y, 400, True),
                              nb_epoch = 30,
                              samples_per_epoch = 40000,
                              verbose = 0)
    p0 += model.predict_generator(generator = batch_generatorp(xtest_mod0, 800, False), val_samples = xtest_mod0.shape[0])
    p1 += model.predict_generator(generator = batch_generatorp(xtest_mod1, 800, False), val_samples = xtest_mod1.shape[0])

p0 /= nbags
p1 /= nbags

p_age_group = np.column_stack([p0, p1])

#-------------------------------------------------------------------------------------------------
# combine predictions
#-------------------------------------------------------------------------------------------------

s1 = p_age_group[:, range(0,6)] * p_gender[:, 0, np.newaxis]  
s2 = p_age_group[:, range(6,12)] * p_gender[:, 1, np.newaxis]  
p_group = np.column_stack([s1, s2])

#-------------------------------------------------------------------------------------------------
# create submission
#-------------------------------------------------------------------------------------------------

col_labels = ('F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+', 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+')       
df = pd.DataFrame(p_group , columns = col_labels)
df['device_id'] = device_id
df = df.set_index('device_id')
df.to_csv(PATH + '/preds/keras_2_test_noevents.csv', index=True, index_label='device_id')
