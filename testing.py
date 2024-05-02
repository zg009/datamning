import multiprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold, KFold, GridSearchCV, RandomizedSearchCV

from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, ndcg_score
import time

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
# from imblearn.combine import SMOTEEN
import imblearn
from joblib import dump, load
import collections

LEFT_TESTING_DATA = './sparse_testing_data.csv'
test_df = pd.read_csv(LEFT_TESTING_DATA, index_col='id')
test_df = test_df.drop(test_df.columns[[0]], axis=1)

LEFT_TRAINING_DATA = './sparse_training_data.csv'
training_df = pd.read_csv(LEFT_TRAINING_DATA, index_col='id')
training_df = training_df.drop(training_df.columns[[0]], axis=1)
wrong_cols = ['booked', 'country_destination']
missing_cols = []

# If column inside test df is not in training, should not be there
for col in test_df.columns:
    if col not in training_df.columns:
        wrong_cols.append(col)

# If column from training in not in testing data, add it
for col in training_df.columns:
    if col not in test_df.columns:
        missing_cols.append(col)

# Make it all zeroes
for col in missing_cols:
    test_df[col] = 0

# Drop all columns that are not in training
test_df = test_df.drop(wrong_cols, axis=1)
print(collections.Counter(test_df.columns.tolist()) == collections.Counter(training_df.columns.tolist()))

# knn = load('knn-classifier.joblib')
knn = load('bag-classifier-mc.joblib')

test_df = test_df[knn.feature_names_in_.tolist()]
print(test_df)

y_pred = knn.predict_proba(test_df)
print(y_pred)

sparse_mc_y = training_df.country_destination

label_encoder = LabelEncoder()

encoded_y = label_encoder.fit_transform(sparse_mc_y)

prediction=[]
final_prediction = []

test_df = test_df.reset_index() 
#df full of id's
idea = test_df['id']

#This gets top 5 predictions and takes top one. Probably better way to do this
for i in range(idea.size):
    prediction.append(label_encoder.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist())  
    final_prediction.append(prediction[i][0])

pf = pd.DataFrame({'id': idea, 'country': final_prediction})
pf.describe()
print(pf)
# file_name = 'k-neighbor-smoteenn-predictions.csv'
file_name = 'bag12-decisiontree-predictions.csv'
# file_name = 'bag40-decisiontree-predictions.csv'


pf.to_csv(file_name, index=False)