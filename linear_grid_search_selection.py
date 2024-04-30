import multiprocessing
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.svm import LinearSVC
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedBaggingClassifier
from joblib import dump

LEFT_TRAINING_DATA = './sparse_training_data.csv'


# binary_classifier()
# without imbalanced sampling {'C': 0.1, 'max_iter': 2000, 'penalty': 'l1'} ~ 0.69
# with random under sampler
# 0.5698981952447655
# {'C': 0.001, 'max_iter': 1500, 'penalty': 'l1'}
# with random over sampler
# 0.5837863177245499
# {'C': 0.001, 'max_iter': 1000, 'penalty': 'l1'}
# with SMOTEEN
# LinearSVC(C=0.1, dual=False, max_iter=2000, penalty='l1')
# 0.791031018492813
# with SMOTEtomek
# 0.6140436553552178
# {'C': 0.001, 'max_iter': 1000, 'penalty': 'l1'}
def binary_classifier():
    sparse = pd.read_csv(LEFT_TRAINING_DATA, index_col='id')
    sparse = sparse.drop(sparse.columns[[0]], axis=1)
    # with sparse, only 170 columns have no nulls for NDF
    sparse_X = sparse.drop(['country_destination', 'booked'], axis=1)
    sparse_binary_y = sparse.booked

    param_grid = {
        'C': [0.001, 0.1, 1],
        'penalty': ['l1', 'l2'],
        'max_iter': [1000, 1500, 2000]
    }

    # X_train, X_test, y_train, y_test = train_test_split(sparse_X, sparse_binary_y, test_size=0.3, random_state=42)
    # rus = RandomUnderSampler(random_state=42)
    # ros = RandomOverSampler(random_state=42)
    smoteen = SMOTEENN(random_state=42)
    # smotetomek = SMOTETomek(random_state=42)
    X_train, y_train = smoteen.fit_resample(sparse_X, sparse_binary_y)

    model = LinearSVC(dual=False)
    # bc = BalancedBaggingClassifier(model, replacement=False, sampling_strategy='auto', random_state=42)
    clf = GridSearchCV(model, param_grid, n_jobs=multiprocessing.cpu_count() - 2, verbose=2)
    clf.fit(X_train, y_train)
    decisions = clf.decision_function(X_train)
    print(decisions)
    # print(clf.cv_results_)
    print(clf.best_estimator_)
    dump(clf.best_estimator_, 'linearsvc_SMOTEEN_binary_classifier.joblib')
    print(clf.best_score_)
    print(clf.best_params_)


sparse = pd.read_csv(LEFT_TRAINING_DATA, index_col='id')
sparse = sparse.drop(sparse.columns[[0]], axis=1)
sparse = sparse.drop(sparse[(sparse['country_destination'] == 'NDF')].index)
# with sparse, only 170 columns have no nulls for NDF
sparse_X = sparse.drop(['country_destination', 'booked'], axis=1)
sparse_y = sparse.country_destination

param_grid = {
    'C': [0.001, 0.1, 1],
    'penalty': ['l1', 'l2'],
    'max_iter': [1000, 2000, 3000]
}
# X_train, X_test, y_train, y_test = train_test_split(sparse_X, sparse_y, test_size=0.3, random_state=42)
smoteen = SMOTEENN(random_state=42)
X_train, y_train = smoteen.fit_resample(sparse_X, sparse_y)

model = LinearSVC(dual=False)
# bc = BalancedBaggingClassifier(model, replacement=False, sampling_strategy='auto', random_state=42)
clf = GridSearchCV(model, param_grid, n_jobs=multiprocessing.cpu_count() - 2, verbose=2)
clf.fit(X_train, y_train)
print(clf.best_estimator_)
dump(clf.best_estimator_, 'linearsvc_smoteen_ovr_classifier.joblib')
print(clf.best_score_)
print(clf.best_params_)
# Linear SVC train_test_split
# LinearSVC(C=0.001, dual=False)
# 0.7018076644974692
# {'C': 0.001, 'max_iter': 1000, 'penalty': 'l2'}