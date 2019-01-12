# coding=utf8

import pandas as pd
import numpy as np

# import xgboost as xgb
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.decomposition import TruncatedSVD

# Create bag-of-apps in character string format
# first by event
# then merge to generate larger bags by device

##################
#   App Labels
##################

print("# Read App Labels")
app_lab = pd.read_csv("./input/app_labels.csv", dtype={"device_id": np.str})
app_lab = app_lab.groupby("app_id")["label_id"].apply(
    lambda x: " ".join(str(s) for s in x)
)

##################
#   App Events
##################
print("# Read App Events")
app_ev = pd.read_csv("./input/app_events.csv", dtype={"device_id": np.str})
app_ev["app_lab"] = app_ev["app_id"].map(app_lab)
app_ev = app_ev.groupby("event_id")["app_lab"].apply(
    lambda x: " ".join(str(s) for s in x)
)

del app_lab

##################
#     Events
##################
print("# Read Events")
events = pd.read_csv("./input/events.csv", dtype={"device_id": np.str})
events["app_lab"] = events["event_id"].map(app_ev)
events = events.groupby("device_id")["app_lab"].apply(
    lambda x: " ".join(str(s) for s in x)
)

del app_ev

##################
#   Phone Brand
##################
print("# Read Phone Brand")
pbd = pd.read_csv("./input/phone_brand_device_model.csv", dtype={"device_id": np.str})
pbd.drop_duplicates("device_id", keep="first", inplace=True)


##################
#  Train and Test
##################
print("# Generate Train and Test")
train = pd.read_csv("./input/gender_age_train.csv", dtype={"device_id": np.str})
train["app_lab"] = train["device_id"].map(events)
train = pd.merge(train, pbd, how="left", on="device_id", left_index=True)

test = pd.read_csv("./input/gender_age_test.csv", dtype={"device_id": np.str})
test["app_lab"] = test["device_id"].map(events)
test = pd.merge(test, pbd, how="left", on="device_id", left_index=True)

del pbd
del events


def get_hash_data(train, test):
    df = pd.concat((train, test), axis=0, ignore_index=True)
    split_len = len(train)

    # TF-IDF Feature
    tfv = TfidfVectorizer(min_df=1)
    df = (
        df[["phone_brand", "device_model", "app_lab"]]
        .astype(np.str)
        .apply(lambda x: " ".join(s for s in x), axis=1)
        .fillna("Missing")
    )
    df_tfv = tfv.fit_transform(df)

    train = df_tfv[:split_len, :]
    test = df_tfv[split_len:, :]
    return train, test


train_device_id = train["device_id"].values
test_device_id = test["device_id"].values

train, test = get_hash_data(train, test)
# train.to_csv("rds\\train_py2.csv", index=False)
# test.to_csv("rds\\test_py2.csv", index=False)

print("# Applying svd & exporting")
svd = TruncatedSVD(n_components=20, n_iter=5)  # Dimensionality reduction. PCA
svd.fit(train)

s_data = svd.transform(train)
t_data = svd.transform(test)
s_data = pd.DataFrame(s_data)
t_data = pd.DataFrame(t_data)

s_data["device_id"] = train_device_id
t_data["device_id"] = test_device_id

s_data.to_csv("rds/train_svd200.csv", delimiter=",")
t_data.to_csv("rds/test_svd200.csv", delimiter=",")
