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

LEFT_TRAINING_DATA = './sparse_training_data.csv'
sparse = pd.read_csv(LEFT_TRAINING_DATA, index_col='id')
sparse = sparse.drop(sparse.columns[[0]], axis=1)
# with sparse, only 170 columns have no nulls for NDF
sparse_X = sparse.drop(['country_destination', 'booked'], axis=1)
sparse_binary_y = sparse.booked
sparse_mc_y = sparse.country_destination
label_encoder = LabelEncoder()
# print(sparse_mc_y)

encoded_y = label_encoder.fit_transform(sparse_mc_y) #Transforming the target variable using labels
print(encoded_y)

print("Doing SMOTEENN")
smoteen = imblearn.combine.SMOTEENN(random_state=42, n_jobs=multiprocessing.cpu_count() - 1)
X_train, y_train = smoteen.fit_resample(sparse_X, encoded_y)

print("Splitting sets")
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(sparse_X, encoded_y, test_size=0.3, random_state=42)
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test, )
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Creating classifier")

# knn = KNeighborsClassifier(n_neighbors = 7, n_jobs=multiprocessing.cpu_count() - 1).fit(X_train, y_train)
# dump(knn, 'knn-classifier.joblib')
knn = load('knn-classifier.joblib')
print("Done loading")
# # accuracy on X_test 
accuracy = knn.score(X_test, y_test) 
print(accuracy) 
  
# # creating a confusion matrix 
# knn_predictions = knn.predict(X_test)  
# cm = confusion_matrix(y_test, knn_predictions) 
# print(cm)

y_pred = knn.predict_proba(X_test)
print(y_pred)
print(y_pred[0])
print("Done predict")
# accuracy = ndcg_score(y_true= y_test,
#                            y_score= y_pred,
#                            k=5, sample_weight=None, ignore_ties=False)
le = LabelEncoder()
y_test = le.fit_transform(y_test)


accuracy = ndcg_score(y_test, y_pred, k=5)

print(accuracy)



prediction=[]
for i in range(20):
     prediction.append(le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist())  

actual = le.inverse_transform(y_test[0:20]).tolist()
print(actual)

pf = pd.DataFrame({'Prediction': prediction, 'Actual': actual, 'Accuracy Score': [accuracy]*20})
pf.describe()
print(pf)
