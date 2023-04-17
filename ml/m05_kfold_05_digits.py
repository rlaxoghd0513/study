
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x,y = load_digits(return_X_y=True)
n_splits = 5
kfold = KFold(n_splits= n_splits, shuffle = True, random_state = 123)

model = RandomForestClassifier()

scores = cross_val_score(model, x,y, cv = kfold, n_jobs=4)

print('acc:', scores,'\n cross_val_score 평균:', round(np.mean(scores),4))

# acc: [0.975      0.97222222 0.97771588 0.97771588 0.98607242] 
#  cross_val_score 평균: 0.9777