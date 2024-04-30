import multiprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold, KFold, GridSearchCV

from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score
import time

from imblearn.combine import SMOTEEN
# TRAINING_DATA = './training_data.csv'
# df = pd.read_csv(TRAINING_DATA, index_col='id')
# df = df.drop(df.columns[[0]], axis=1)
# X = df.drop(['country_destination'], axis=1)
# y = df.country_destination

LEFT_TRAINING_DATA = './sparse_training_data.csv'
sparse = pd.read_csv(LEFT_TRAINING_DATA, index_col='id')
sparse = sparse.drop(sparse.columns[[0]], axis=1)
# with sparse, only 170 columns have no nulls for NDF
sparse_X = sparse.drop(['country_destination', 'booked'], axis=1)
sparse_binary_y = sparse.booked
sparse_mc_y = sparse.country_destination
X_train, X_test, y_train, y_test = train_test_split(sparse_X, sparse_binary_y, test_size=0.3, random_state=42)

# LEFT_TESTING_DATA = './sparse_testing_data.csv'
# testing = pd.read_csv(LEFT_TESTING_DATA, index_col='id')
# testing_X = testing.drop(testing.columns[[0]], axis=1)

# p_grid = {
#     'penalty': ['l1', 'l2'],
#     # 'C': [1, 10, 100]
# }

# svc = LinearSVC(dual=False, max_iter=3000)

# inner_cv = KFold(n_splits=4, shuffle=True)
# outer_cv = KFold(n_splits=4, shuffle=True)

# clf = GridSearchCV(estimator=svc, cv=outer_cv, param_grid=p_grid)
# clf.fit(sparse_X, sparse_binary_y)
# print(clf.best_score_)

#  this seems better
# clf = GridSearchCV(estimator=svc, param_grid=p_grid, cv=inner_cv)
# nested_score = cross_val_score(clf, X=sparse_X, y=sparse_binary_y, cv=outer_cv)
# print(nested_score)
# print(cpu_count())
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'max_iter': [1500, 2000, 2500, 3000]
}
X_train, X_test, y_train, y_test = train_test_split(sparse_X, sparse_binary_y, test_size=0.3, random_state=42)
start_time = time.time()
model = LinearSVC(dual=False)
clf = GridSearchCV(model, param_grid, n_jobs=multiprocessing.cpu_count() - 2, verbose=2)
clf.fit(X_train, y_train)
decisions = clf.decision_function(X_train)
print(decisions)
# print(clf.cv_results_)
print(clf.best_estimator_)
print(clf.best_score_)
print(clf.best_params_)
# print(int(cpu_count()) - 2) # type: ignore
# ovo = OneVsRestClassifier(model, n_jobs=int(cpu_count()) - 2) # type: ignore
# print(X_train.shape, X_test.shape)
# model.fit(X_train, y_train)
# y_hat = model.predict(X_test)
# score = accuracy_score(y_hat, y_test)
# print('SVC score', score)
# print("--- %s seconds ---" % (time.time() - start_time))