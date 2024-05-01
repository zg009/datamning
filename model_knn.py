import multiprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold, KFold, GridSearchCV

from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import time

from sklearn.neighbors import KNeighborsClassifier 

# from imblearn.combine import SMOTEEN
import imblearn

LEFT_TRAINING_DATA = './sparse_training_data.csv'
sparse = pd.read_csv(LEFT_TRAINING_DATA, index_col='id')
sparse = sparse.drop(sparse.columns[[0]], axis=1)
# with sparse, only 170 columns have no nulls for NDF
sparse_X = sparse.drop(['country_destination', 'booked'], axis=1)
sparse_binary_y = sparse.booked
sparse_mc_y = sparse.country_destination

smoteen = imblearn.combine.SMOTEENN(random_state=42)
X_train, y_train = smoteen.fit_resample(sparse_X, sparse_binary_y)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
# accuracy on X_test 
accuracy = knn.score(X_test, y_test) 
print(accuracy) 
  
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
cm = confusion_matrix(y_test, knn_predictions) 
print(cm)