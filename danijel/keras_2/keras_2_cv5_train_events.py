
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

## path of working directory
PATH = '../danijel'

## read folds
folds = pd.read_csv(PATH + '/folds/folds_5.csv')['fold'].values

## device ids
device_id_train = pd.read_csv(PATH + '/input/for_py_train_device_id_all.csv')['x'].values

## read xtrain
triples = pd.read_csv(PATH + '/input/for_py_xtrain_triples_all.csv')

nrow = np.max(triples['row']) + 1
ncol = np.max(triples['col']) + 1

xtrain = sparse.csr_matrix(
    (triples['x'], (triples['row'], triples['col'])), shape = (nrow, ncol))

## response
y = pd.read_csv(PATH + '/input/for_py_group_all.csv')['x'].values

## device ids with events
has_events = np.asarray(xtrain[:,0].todense())[:,0]
sel = [idx for idx, event in enumerate(has_events) if event == 1]

## subset
folds = folds[sel]
device_id_train = device_id_train[sel]
xtrain = xtrain[sel]
y = y[sel]

## neural net
def nn_model():
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
  return(model)

## object to store preds
p_group = np.zeros((xtrain.shape[0], 12))

## cv params
nfolds = np.max(folds)
nbags = 5

## cv index
for i in range(1, nfolds+1):
  #--------------------------------------------------------------------------------------------------------------
  # neural net - group 
  #--------------------------------------------------------------------------------------------------------------
  inTr = [idx for idx, fold in enumerate(folds) if fold != i]
  inTe = [idx for idx, fold in enumerate(folds) if fold == i]
  ## train data
  xtr = xtrain[inTr]
  ytr = y[inTr]
  ## validation data
  xval = xtrain[inTe]
  yval = y[inTe]
  ## pred object
  pred = np.zeros((xval.shape[0], 12))
  for j in range(nbags):
    model = nn_model()
    ## training
    model.fit_generator(generator = batch_generator(xtr, ytr, 200, True),
                        nb_epoch = 80,
                        samples_per_epoch = 1000,
                        #validation_data = (xval.todense(), yval), 
                        verbose = 0)
    ## prediction
    pred += model.predict_generator(generator = batch_generatorp(xval, 800, False), val_samples = xval.shape[0])
  ## average predictions
  pred /= nbags
  p_group[inTe] = pred
  score = log_loss(yval, pred)
  print('Iter ', i, '-', score, '\n')


log_loss(y, p_group)
# 1.9199875656676091

## save cv predictions
col_labels = ('F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+', 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+')       
df = pd.DataFrame(p_group, columns = col_labels)
df['device_id'] = device_id_train
df = df.set_index('device_id')
df.to_csv(PATH + '/preds/keras_2_cv5_train_events.csv', index = True, index_label = 'device_id')





