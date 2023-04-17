import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
print(train_csv.shape) #(652, 9)

test_csv = pd.read_csv(path+'test.csv', index_col=0)
print(test_csv.shape) #(116, 8)

print(train_csv.info()) #non-null

x = train_csv.drop(['Outcome'], axis=1)
print(x) #[652 rows x 8 columns]

y = train_csv['Outcome']

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state=123)

model = RandomForestClassifier()

scores = cross_val_score(model, x, y, cv=kfold)
print('acc:', scores, '\n cross_val_score평균:', round(np.mean(scores),4))

#acc: [0.7480916  0.77862595 0.71538462 0.80769231 0.79230769] 
#  cross_val_score평균: 0.7684