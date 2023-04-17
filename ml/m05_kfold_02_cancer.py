import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

x, y = load_breast_cancer(return_X_y=True)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state=12)

model = RandomForestClassifier()

scores = cross_val_score(model, x, y, cv=kfold)

print('acc:', scores, '\n cross_val_score평균:', round(np.mean(scores),4))

# acc: [0.93859649 0.92982456 0.99122807 1.         0.98230088] 
# cross_val_score평균: 0.9684