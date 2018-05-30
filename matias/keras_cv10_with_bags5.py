import numpy as np
seed = 7
np.random.seed(seed)

import pandas as pd
import pandas.core.algorithms as algos

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
#from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import os
from scipy.sparse import csr_matrix, hstack

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers.advanced_activations import PReLU

# load dataset

datadir = './input'
#datadir = '/home/username/projects/talkingData/input'
#datadir = 'C:\\mthayer\\competition\\talkingData\\input'
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'), index_col = 'device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir,'events.csv'),  parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), usecols=['event_id','app_id','is_active'], dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))


##Load the CV split.
#datadir = 'C:\\mthayer\\competition\\talkingData\\tdCode'
folds=pd.read_csv(os.path.join(datadir,"folds_10.csv"), index_col='device_id')

##Reorder train and cv so the device ids match afterwards
gatrain=gatrain.sort_index()
folds=folds.sort_index()

print("validation, must be zero!", sum(gatrain.index!=folds.index))

####Phone brand
#As preparation I create two columns that show which train or test set row a particular device_id belongs to.

gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

# A sparse matrix of features can be constructed in various ways. I use this constructor:
# csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
# where ``data``, ``row_ind`` and ``col_ind`` satisfy the
# relationship ``a[row_ind[k], col_ind[k]] = data[k]``
#
# It lets me specify which values to put into which places in a sparse matrix. For phone brand data the data array will be all ones,
# row_ind will be the row number of a device and col_ind will be the number of brand.

brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]),
                       (gatrain.trainrow, gatrain.brand)))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]),
                       (gatest.testrow, gatest.brand)))
print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))

# Device model
m = phone.phone_brand.str.cat(phone.device_model)

modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']

Xtr_model = csr_matrix((np.ones(gatrain.shape[0]),
                       (gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]),
                       (gatest.testrow, gatest.model)))
print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))

###############################################
###MT work: Frequency term:
# model_freq = (phone
#                 .groupby(['model'])['model'].count()
#                 .to_frame())
# model_freq.columns.values[0]='model_freq'

model_freq = phone["model"].value_counts().to_frame()
mf_encoder = LabelEncoder().fit(model_freq.model)
model_freq['model_freq']=mf_encoder.transform(model_freq['model'])
model_freq= model_freq.drop("model", 1)

gatrain=gatrain.merge(model_freq, how='left', left_on="model", right_index=True)
gatest=gatest.merge(model_freq, how='left', left_on="model", right_index=True)
gatest["model_freq"]=gatest["model_freq"].fillna(1) # fill not found frequencies with 1


Xtr_model_freq = csr_matrix((np.ones(gatrain.shape[0]),
                       (gatrain.trainrow, gatrain["model_freq"])))
Xte_model_freq = csr_matrix((np.ones(gatest.shape[0]),
                       (gatest.testrow, gatest["model_freq"])))

print('Model frequency features: train shape {}, test shape {}'.format(Xtr_model_freq.shape, Xte_model_freq.shape))

brand_freq = phone["brand"].value_counts().to_frame()
bf_encoder = LabelEncoder().fit(brand_freq.brand)
brand_freq['brand_freq']=bf_encoder.transform(brand_freq['brand'])
brand_freq= brand_freq.drop("brand", 1)

brand_freq.columns.values[0]='brand_freq'
gatrain=gatrain.merge(brand_freq, how='left', left_on="brand", right_index=True)
gatest=gatest.merge(brand_freq, how='left', left_on="brand", right_index=True)
gatest["brand_freq"]=gatest["brand_freq"].fillna(1) # fill not found frequencies with 1

Xtr_brand_freq = csr_matrix((np.ones(gatrain.shape[0]),
                       (gatrain.trainrow, gatrain.brand_freq)))

Xte_brand_freq = csr_matrix((np.ones(gatest.shape[0]),
                       (gatest.testrow, gatest.brand_freq)))

print('Brand frequency features: train shape {}, test shape {}'.format(Xtr_brand_freq.shape, Xte_brand_freq.shape))


#############################################

# Installed apps features
# For each device I want to mark which apps it has installed. So I'll have as many feature columns as there are distinct apps.
# Apps are linked to devices through events. So I do the following:
# merge device_id column from events table to app_events
# group the resulting dataframe by device_id and app and aggregate
# merge in trainrow and testrow columns to know at which row to put each device in the features matrix

appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['max'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())

d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)),
                      shape=(gatrain.shape[0],napps))
d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)),
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))

# App labels features
# These are constructed in a way similar to apps features by merging app_labels with the deviceapps dataframe we created above.
applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
applabels['app'] = appencoder.transform(applabels.app_id)
labelencoder = LabelEncoder().fit(applabels.label_id)
applabels['label'] = labelencoder.transform(applabels.label_id)
nlabels = len(labelencoder.classes_)

devicelabels = (deviceapps[['device_id','app']]
                .merge(applabels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
devicelabels.head()

d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)),
                      shape=(gatrain.shape[0],nlabels))
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)),
                      shape=(gatest.shape[0],nlabels))
print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))

events_cout = (events.groupby('device_id')['timestamp'].agg(['size'])
                    .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                    .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                    .reset_index())
events_cout.size = (np.log((events_cout['size'])))
events_cout.size = events_cout.size/events_cout.size.max()

d = events_cout.dropna(subset=['trainrow'])
Xtr_eventsize = csr_matrix((d.iloc[:,1], (d.trainrow, np.zeros(d.shape[0]))),
                      shape=(gatrain.shape[0],1))

d = events_cout.dropna(subset=['testrow'])
Xte_eventsize = csr_matrix((d.iloc[:,1], (d.testrow, np.zeros(d.shape[0]))),
                      shape=(gatest.shape[0],1))
print('Labels data: train shape {}, test shape {}'.format(Xtr_eventsize.shape, Xte_eventsize.shape))

events['hour'] = events.timestamp.apply(lambda x: x.hour)
events_cout_hourofday = (events.groupby(['device_id','hour'])['hour'].agg(['size'])
                    .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                    .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                    .reset_index())
d = events_cout_hourofday.dropna(subset=['trainrow'])
Xtr_event_on_hourofday = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.hour)),
                      shape=(gatrain.shape[0],d.hour.nunique()))

d = events_cout_hourofday.dropna(subset=['testrow'])
Xte_event_on_hourofday = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.hour)),
                      shape=(gatest.shape[0],d.hour.nunique()))
print('Labels data: train shape {}, test shape {}'.format(Xtr_event_on_hourofday.shape, Xte_event_on_hourofday.shape))

###########################
# Create bag-of-apps in character string format
# first by event
# then merge to generate larger bags by device

##################
#   App Labels
##################

print("# Read App Labels")
app_lab = pd.read_csv("./input/app_labels.csv")
app_lab = app_lab.groupby("app_id")["label_id"].apply(
    lambda x: " ".join(str(s) for s in x))

##################
#   App Events
##################
print("# Read App Events")
app_ev = pd.read_csv("./input/app_events.csv")
app_ev["app_lab"] = app_ev["app_id"].map(app_lab)
app_ev = app_ev.groupby("event_id")["app_lab"].apply(
    lambda x: " ".join(str(s) for s in x))

del app_lab

##################
#     Events
##################
print("# Read Events")
events = pd.read_csv("./input/events.csv")
events["app_lab"] = events["event_id"].map(app_ev)
events = events.groupby("device_id")["app_lab"].apply(
    lambda x: " ".join(str(s) for s in x))

del app_ev

##################
#   Phone Brand
##################
print("# Read Phone Brand")
pbd = pd.read_csv("./input/phone_brand_device_model.csv")
pbd.drop_duplicates('device_id', keep='first', inplace=True)


##################
#  Train and Test
##################
print("# Generate Train and Test")
#train = pd.read_csv("./input/gender_age_train.csv")
train = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), index_col='device_id')

train["dev_id"]=train.index
train["app_lab"] = train["dev_id"].map(events)
# train = pd.merge(train, pbd, how='left',
#                  on='device_id', left_index=True)
train=pd.merge(train, pbd, how='left', left_index=True, right_on="device_id")
train.index=train["dev_id"]

train=train.sort_index()

print("Before hash: must be zero: ", sum(train.index != gatrain.index))
#print("Before hash: must be zero: ", sum(train["dev_id"] != gatrain.index))


# test = pd.read_csv("./input/gender_age_test.csv",
#                    dtype={'device_id': np.str})
test = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'), index_col='device_id')
test["dev_id"]=test.index

test["app_lab"] = test["dev_id"].map(events)
# test = pd.merge(test, pbd, how='left',
#                 on='device_id', left_index=True)
test=pd.merge(test, pbd, how='left', left_index=True, right_on="device_id")

del pbd
del events


####Phone brand
#
def get_hash_data(train, test):
    df = pd.concat((train, test), axis=0, ignore_index=True)
    split_len = len(train)

    # TF-IDF Feature
    tfv = TfidfVectorizer(min_df=1)
    df = df[["phone_brand", "device_model", "app_lab"]].astype(np.str).apply(
        lambda x: " ".join(s for s in x), axis=1).fillna("Missing")
    df_tfv = tfv.fit_transform(df)

    train = df_tfv[:split_len, :]
    test = df_tfv[split_len:, :]
    return train, test

trainrow = np.arange(train.shape[0])
testrow = np.arange(test.shape[0])
superrow= np.arange(train.shape[0]+ test.shape[0])

train_device_id = train["device_id"].values
test_device_id = test["device_id"].values

train_bag, test_bag = get_hash_data(train,test)

del train
del test

print("After hash: must be zero: ", sum(train_device_id != gatrain.index))


#train.to_csv("rds\\train_py2.csv", index=False)
#test.to_csv("rds\\test_py2.csv", index=False)
#
# print("# Applying svd & exporting")
# svd = TruncatedSVD(n_components = 20,n_iter=5) #Dimensionality reduction. PCA
# svd.fit(train)
#
# s_data = svd.transform(train)
# t_data = svd.transform(test)
# s_data = pd.DataFrame(s_data)
# t_data = pd.DataFrame(t_data)
#
# s_data['device_id'] = train_device_id
# t_data['device_id'] = test_device_id

#s_data.to_csv("rds/train_svd200.csv", delimiter=",")
#t_data.to_csv("rds/test_svd200.csv",  delimiter=",")

# super=pd.concat((s_data, t_data), axis=0, ignore_index=True)
# split_len = len(s_data)


#s_data[0].values
#import matplotlib.pyplot as plt
#plt.hist(super[0])
#plt.hist(super[3])
#plt.show()
#super[0].max()

# super["q1"]=pd.cut(super[0], super[0].quantile(np.linspace(0, 1, 11)), labels=range(1,11), include_lowest=True)
# super["q2"]=pd.cut(super[1], super[1].quantile(np.linspace(0, 1, 11)), labels=range(1,11), include_lowest=True)
# super["q3"]=pd.cut(super[2], super[2].quantile(np.linspace(0, 1, 11)), labels=range(1,11), include_lowest=True)
# super["q4"]=pd.cut(super[3], super[3].quantile(np.linspace(0, 1, 11)), labels=range(1,11), include_lowest=True)
# super["q5"]=pd.cut(super[4], super[4].quantile(np.linspace(0, 1, 11)), labels=range(1,11), include_lowest=True)

# super["q1"] = pd.tools.tile._bins_to_cuts(super[0], include_lowest=True, labels=range(1,16))
# super["q2"] = pd.tools.tile._bins_to_cuts(super[1], algos.quantile(np.unique(super[1]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q3"] = pd.tools.tile._bins_to_cuts(super[2], algos.quantile(np.unique(super[2]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q4"] = pd.tools.tile._bins_to_cuts(super[3], algos.quantile(np.unique(super[3]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q5"] = pd.tools.tile._bins_to_cuts(super[4], algos.quantile(np.unique(super[4]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q6"] = pd.tools.tile._bins_to_cuts(super[5], algos.quantile(np.unique(super[5]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q7"] = pd.tools.tile._bins_to_cuts(super[6], algos.quantile(np.unique(super[6]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q8"] = pd.tools.tile._bins_to_cuts(super[7], algos.quantile(np.unique(super[7]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q9"] = pd.tools.tile._bins_to_cuts(super[8], algos.quantile(np.unique(super[8]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q10"] = pd.tools.tile._bins_to_cuts(super[9], algos.quantile(np.unique(super[9]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q11"] = pd.tools.tile._bins_to_cuts(super[10], algos.quantile(np.unique(super[10]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q12"] = pd.tools.tile._bins_to_cuts(super[11], algos.quantile(np.unique(super[11]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q13"] = pd.tools.tile._bins_to_cuts(super[12], algos.quantile(np.unique(super[12]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q14"] = pd.tools.tile._bins_to_cuts(super[13], algos.quantile(np.unique(super[13]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q15"] = pd.tools.tile._bins_to_cuts(super[14], algos.quantile(np.unique(super[14]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q16"] = pd.tools.tile._bins_to_cuts(super[15], algos.quantile(np.unique(super[15]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q17"] = pd.tools.tile._bins_to_cuts(super[16], algos.quantile(np.unique(super[16]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q18"] = pd.tools.tile._bins_to_cuts(super[17], algos.quantile(np.unique(super[17]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q19"] = pd.tools.tile._bins_to_cuts(super[18], algos.quantile(np.unique(super[18]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))
# super["q20"] = pd.tools.tile._bins_to_cuts(super[19], algos.quantile(np.unique(super[19]), np.linspace(0, 1, 16)), include_lowest=True, labels=range(1,16))

# len(super["q1"])
# su_model_q1 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q1"])))
# su_model_q2 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q2"])))
# su_model_q3 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q3"])))
# su_model_q4 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q4"])))
# su_model_q5 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q5"])))
# su_model_q6 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q6"])))
# su_model_q7 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q7"])))
# su_model_q8 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q8"])))
# su_model_q9 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q9"])))
# su_model_q10 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q10"])))
# su_model_q11 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q11"])))
# su_model_q12 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q12"])))
# su_model_q13 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q13"])))
# su_model_q14 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q14"])))
# su_model_q15 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q15"])))
# su_model_q16 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q16"])))
# su_model_q17 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q17"])))
# su_model_q18 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q18"])))
# su_model_q19 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q19"])))
# su_model_q20 = csr_matrix((np.ones(super.shape[0]), (superrow, super["q20"])))


# tr_model_q1= su_model_q1[:split_len]; te_model_q1= su_model_q1[split_len:]
# tr_model_q2= su_model_q2[:split_len]; te_model_q2= su_model_q2[split_len:]
# tr_model_q3= su_model_q3[:split_len]; te_model_q3= su_model_q3[split_len:]
# tr_model_q4= su_model_q4[:split_len]; te_model_q4= su_model_q4[split_len:]
# tr_model_q5= su_model_q5[:split_len]; te_model_q5= su_model_q5[split_len:]
# tr_model_q6= su_model_q6[:split_len]; te_model_q6= su_model_q6[split_len:]
# tr_model_q7= su_model_q7[:split_len]; te_model_q7= su_model_q7[split_len:]
# tr_model_q8= su_model_q8[:split_len]; te_model_q8= su_model_q8[split_len:]
# tr_model_q9= su_model_q9[:split_len]; te_model_q9= su_model_q9[split_len:]
# tr_model_q10= su_model_q10[:split_len]; te_model_q10= su_model_q10[split_len:]
# tr_model_q11= su_model_q11[:split_len]; te_model_q11= su_model_q11[split_len:]
# tr_model_q12= su_model_q12[:split_len]; te_model_q12= su_model_q12[split_len:]
# tr_model_q13= su_model_q13[:split_len]; te_model_q13= su_model_q13[split_len:]
# tr_model_q14= su_model_q14[:split_len]; te_model_q14= su_model_q14[split_len:]
# tr_model_q15= su_model_q15[:split_len]; te_model_q15= su_model_q15[split_len:]
# tr_model_q16= su_model_q16[:split_len]; te_model_q16= su_model_q16[split_len:]
# tr_model_q17= su_model_q17[:split_len]; te_model_q17= su_model_q17[split_len:]
# tr_model_q18= su_model_q18[:split_len]; te_model_q18= su_model_q18[split_len:]
# tr_model_q19= su_model_q19[:split_len]; te_model_q19= su_model_q19[split_len:]
# tr_model_q20= su_model_q20[:split_len]; te_model_q20= su_model_q20[split_len:]
# Concatenate all features

Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_brand_freq, Xtr_model_freq, Xtr_app, Xtr_label,Xtr_eventsize,Xtr_event_on_hourofday,
                 train_bag), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_brand_freq, Xte_model_freq, Xte_app, Xte_label,Xte_eventsize,Xte_event_on_hourofday,
                 test_bag), format='csr')

print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))

# Reduce dimensionality
indices = np.nonzero(Xtrain)
columns_non_unique = indices[1]
unique_columns = sorted(set(columns_non_unique))
Xtrain=Xtrain.tocsc()[:,unique_columns]
Xtest=Xtest.tocsc()[:,unique_columns]

print('All features after dimensionality reduction: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))

# TIP: Ejemplo accesando a los valores_
#value=Xtr_eventsize.data
#column_index =Xtr_eventsize.indices
#row_pointers =Xtr_eventsize.indptr


#################
# Start modeling
#################

np.random.seed(seed)

targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)

##Keras stuff
dummy_y = np_utils.to_categorical(y) ## Funcion de Keras!

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

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dropout(0.4, input_shape=(Xtrain.shape[1],)))
    model.add(Dense(75))
    model.add(PReLU())
    model.add(Dropout(0.30))
    model.add(Dense(50, init='normal', activation='tanh'))
    model.add(PReLU())
    model.add(Dropout(0.20))

    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

model=baseline_model()
#End of keras stuff

#Logistic regression >
clf1=LogisticRegression(C=0.019, multi_class='multinomial',solver='lbfgs')


#Create predictions repository:
pred = np.zeros((y.shape[0],nclasses*2))
pred_test = np.zeros((gatest.shape[0],nclasses*2))
n_folds=len(folds["fold"].unique())

for fold_id in xrange(1, n_folds + 1):
    #fold_id=1
    train_id=folds["fold"].values!=fold_id
    valid_id = folds["fold"].values == fold_id

    Xtr, Ytr = Xtrain[train_id, :], y[train_id]
    Xva, Yva = Xtrain[valid_id, :], y[valid_id]

    #Fitting logistic regression
    clf1.fit(Xtr, Ytr)
    pred[valid_id,0:12] = clf1.predict_proba(Xva)
    pred_test[:,0:12] = pred_test[:,0:12] + clf1.predict_proba(Xtest)

    score_val=log_loss(Yva, pred[valid_id, 0:12])
    print("Logistic logloss for fold {} is {}". format(fold_id, score_val))

    Ytr_dum, Yva_dum =dummy_y[train_id], dummy_y[valid_id]
    ## Fitting Keras!
    model = baseline_model()
    fit = model.fit_generator(generator=batch_generator(Xtr, Ytr_dum, 381, True),
                              nb_epoch=20,
                              samples_per_epoch=176*381, verbose=2,
                              validation_data=(Xva.todense(), Yva_dum)
                              )

    # evaluate the model
    pred[valid_id, 12:25] = model.predict_generator(generator=batch_generatorp(Xva, 400, False), val_samples=Xva.shape[0])
    pred_test[:, 12:25] = pred_test[:, 12:25] + \
                         model.predict_generator(generator=batch_generatorp(Xtest, 400, False), val_samples=Xtest.shape[0])

    score_val = log_loss(Yva, pred[valid_id, 12:25])
    print("Keras logloss for fold {} is {}".format(fold_id, score_val))

print("## Enf of folds work --------")

col_names=np.concatenate((targetencoder.classes_, targetencoder.classes_), axis=0)

##Averaging predictions for all folds in the test set
pred_test=pred_test/float(n_folds)

#Scaling to 1-0 probs
# for i in xrange(Xtrain.shape[0]):
#     if (sum(pred[i, 0:12]) == 0 | (sum(pred[i, 12:25]) == 0)):
#         print("Error! Alerta, alerta, sum of predictions is zero!")
#         break; exit()
#     else:
#         pred[i, 0:12] = pred[i, 0:12] / float(sum(pred[i, 0:12]))
#         pred[i, 12:25] = pred[i, 12:25] / float(sum(pred[i, 12:25]))
#
# sum(pred[6, 12:25])
# sum(pred_test[6, 12:25])
# for i in xrange(Xtest.shape[0]):
#     if (sum(pred_test[i, 12:25]) == 0):
#         print("Error! Alerta, alerta, sum of predictions is zero!")
#         break; exit()
#     else:
#         pred_test[i, 12:25] = pred_test[i, 12:25] / sum(pred_test[i, 12:25])
#
#     if (sum(pred_test[i, 0:12]) == 0):
#         print("Error! Alerta, alerta, sum of predictions is zero!")
#         break;
#         exit()
#     else:
#         pred_test[i, 0:12] = pred_test[i, 0:12] / sum(pred_test[i, 0:12])
#

score_val=log_loss(y, pred[:,0:12])
print("Logistic: logloss for {} folds is {}". format(n_folds, score_val))


score_val=log_loss(y, pred[:,12:25])
print("Keras: logloss for {} folds is {}". format(n_folds, score_val))


pred_train_df = pd.DataFrame(pred, index = gatrain.index, columns=col_names)
pred_test_df = pd.DataFrame(pred_test, index = gatest.index, columns=col_names)

pred_train_df.to_csv('preds/keras_pred_train_bags5_20160819.csv', index=True, index_label='device_id')
pred_test_df.to_csv('preds/keras_pred_test_bags_SVD20_20160819.csv', index=True, index_label='device_id')

print(pred_test_df.head(1))

#generate prediction:
# submission = pd.DataFrame(pred_test[:,0:12], index = gatest.index, columns=targetencoder.classes_)
# submission.to_csv('/home/username/projects/talkingData/keras_cv10_plus_regression_80_reg.csv',index=True)

submission = pd.DataFrame(pred_test[:,12:25], index = gatest.index, columns=targetencoder.classes_)
submission.to_csv('preds/keras_cv10_with_bags5.csv',index=True)



