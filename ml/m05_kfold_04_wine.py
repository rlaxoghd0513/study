import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x,y = load_wine(return_X_y=True)
n_splits = 5
kfold = KFold(n_splits= n_splits, shuffle = True, random_state = 123)

model = RandomForestClassifier()

scores = cross_val_score(model, x,y, cv = kfold, n_jobs=4) #n_jobs 는cpu 코어 사용갯수

print('acc:', scores,'\n cross_val_score 평균:', round(np.mean(scores),4))

# acc: [1.         1.         0.97222222 1.         0.94285714] 
#  cross_val_score 평균: 0.983