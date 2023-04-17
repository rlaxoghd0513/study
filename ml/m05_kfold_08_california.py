
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x,y = fetch_california_housing(return_X_y=True)
n_splits = 5
kfold = KFold(n_splits= n_splits, shuffle = True, random_state = 123)

model = RandomForestClassifier()

scores = cross_val_score(model, x,y, cv = kfold, n_jobs=4)

print('acc:', scores,'\n cross_val_score 평균:', round(np.mean(scores),4))