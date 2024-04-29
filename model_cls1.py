import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold, KFold, GridSearchCV

from sklearn.svm import SVC, LinearSVC
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

p_grid = {
    'penalty': ['l1', 'l2'],
    'loss': ['hinge', 'squared_hinge'],
    'C': [1, 10, 100],
    'max_iter': [1500, 2000, 2500]
}

svc = LinearSVC(dual=False)
NUM_TRIALS = 6
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

for i in range(NUM_TRIALS):
    print('training... loop', i)
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    
    clf = GridSearchCV(estimator=svc, cv=outer_cv, param_grid=p_grid)
    clf.fit(X_train, y_train)
    non_nested_scores[i] = clf.best_score_
    
    clf = GridSearchCV(estimator=svc, param_grid=p_grid, cv=inner_cv)
    nested_score = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv)
    
    nested_scores[i] = nested_score.mean()


score_difference = non_nested_scores - nested_scores

print("Avg diff of {:6f} with std dev of {:6f}.".format(score_difference.mean(), score_difference.std()))