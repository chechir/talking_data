
## load libraries
import numpy as np

## set seed for reproducability
seed = 123
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

## read train features
triples_train = pd.read_csv(PATH + '/input/for_py_xtrain_triples_all.csv')

nrow = np.max(triples_train['row']) + 1
ncol = np.max(triples_train['col']) + 1

xtrain = sparse.csr_matrix(
    (triples_train['x'], (triples_train['row'], triples_train['col'])), shape = (nrow, ncol))

## read test features
triples_test = pd.read_csv(PATH + '/input/for_py_xtest_triples_all.csv')

nrow = np.max(triples_test['row']) + 1
ncol = np.max(triples_test['col']) + 1

xtest = sparse.csr_matrix(
    (triples_test['x'], (triples_test['row'], triples_test['col'])), shape = (nrow, ncol))

## clear workspace
del(triples_train, triples_test)

## neural net
def nn_model():
    model = Sequential()
    model.add(Dense(200, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(100, init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init = 'he_normal'))
    model.add(Activation('softmax'))
    adagrad = Adagrad(lr = 0.005, epsilon = 1e-08)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = adagrad, metrics = ['accuracy'])
    return(model)

nbags = 10
y = pd.read_csv(PATH + '/input/for_py_group_all.csv')['x'].values
p_group = np.zeros((xtest.shape[0], 12))

for i in range(nbags):
    print('Iter', i, '\n')
    model = nn_model()
    ## training
    model.fit_generator(generator = batch_generator(xtrain, y, 200, True),
                        nb_epoch = 200,
                        samples_per_epoch = 1000,
                        verbose = 0)
    ## prediction
    p_group += model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])



## average prediction
p_group /= nbags

## save predictions
col_labels = ('F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+', 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+')       
df = pd.DataFrame(p_group, columns = col_labels)
device_id_test = pd.read_csv(PATH + '/input/for_py_test_device_id_all.csv')['x'].values
df['device_id'] = device_id_test
df = df.set_index('device_id')
df.to_csv(PATH + '/preds/keras_1_test.csv', index = True, index_label = 'device_id')
