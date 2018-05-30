
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

y = pd.read_csv(PATH + '/input/for_py_group_all.csv')['x'].values
device_id_test = pd.read_csv(PATH + '/input/for_py_test_device_id_all.csv')['x'].values
triples_tr = pd.read_csv(PATH + '/input/for_py_xtrain_triples_all.csv')
triples_te = pd.read_csv(PATH + '/input/for_py_xtest_triples_all.csv')

## xtrain
nrow = np.max(triples_tr['row']) + 1
ncol = np.max(triples_tr['col']) + 1

xtrain = sparse.csr_matrix((triples_tr['x'], (triples_tr['row'], triples_tr['col'])), shape = (nrow, ncol))

## xtest
nrow = np.max(triples_te['row']) + 1
ncol = np.max(triples_te['col']) + 1

xtest = sparse.csr_matrix((triples_te['x'], (triples_te['row'], triples_te['col'])), shape = (nrow, ncol))

del(triples_tr, triples_te)

## device ids with events
has_events_tr = np.asarray(xtrain[:,0].todense())[:,0]
has_events_te = np.asarray(xtest[:,0].todense())[:,0]
sel_tr = [idx for idx, event in enumerate(has_events_tr) if event == 1]
sel_te = [idx for idx, event in enumerate(has_events_te) if event == 1]

## subset
xtrain = xtrain[sel_tr]
y = y[sel_tr]
xtest = xtest[sel_te]
device_id_test = device_id_test[sel_te]

#-------------------------------------------------------------------------------------------------
# training
#-------------------------------------------------------------------------------------------------

nbags = 10
scores = np.zeros((xtest.shape[0], 12))

for i in range(nbags):
    ## neural net
    model = Sequential()
    model.add(Dense(200, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(100, init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init = 'he_normal', activation = 'softmax'))
    adagrad = Adagrad(lr = 0.005, epsilon = 1e-08)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = adagrad, metrics = ['accuracy'])
    ## train model
    fit = model.fit_generator(generator = batch_generator(xtrain, y, 200, True),
                              nb_epoch = 80,
                              samples_per_epoch = 1000,
                              verbose = 0)
    ## predict
    scores += model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])

scores /= nbags

## create submission
col_labels = ('F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+', 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+')       
df = pd.DataFrame(scores , columns = col_labels)
df['device_id'] = device_id_test
df = df.set_index('device_id')
df.to_csv(PATH + '/preds/keras_2_test_events.csv', index=True, index_label='device_id')


