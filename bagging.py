import multiprocessing
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import imblearn
from joblib import dump, load

LEFT_TRAINING_DATA = './sparse_training_data.csv'
sparse = pd.read_csv(LEFT_TRAINING_DATA, index_col='id')
sparse = sparse.drop(sparse.columns[[0]], axis=1)

sparse_mc_y = sparse.country_destination
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(sparse_mc_y) #Transforming the target variable using labels
print(f'{sparse_mc_y[0]} : {encoded_y[0]}')

sparse_X = sparse.drop(['country_destination', 'booked'], axis=1)
print(sparse_X)

# print("smote time")
# smoteen = imblearn.combine.SMOTEENN(random_state=42, n_jobs=multiprocessing.cpu_count() - 1)
# X_train, y_train = smoteen.fit_resample(sparse_X, encoded_y)

print("split time")
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(sparse_X, encoded_y, test_size=0.3, random_state=42)

print("classifier time")
#Originally 10 estimators
# Now trying 12
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=12, max_samples=0.5, max_features=0.5, n_jobs=multiprocessing.cpu_count() - 1)
bagging.fit(X_train, y_train)
print(bagging.score(X_test,y_test))

dump(bagging, 'bag-classifier-mc.joblib')