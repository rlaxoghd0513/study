import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)

test_csv = pd.read_csv(path+'test.csv', index_col=0)

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']
print(x.shape, y.shape) #(652, 8) (652,)

x_train, x_test, y_train, y_test = train_test_split(x,y,shuffle=True, random_state=333, train_size=0.8)

model = RandomForestClassifier(random_state = 333)
model.fit(x_train, y_train)
result = model.score(x_test, y_test)


lda = LinearDiscriminantAnalysis()
x1 = lda.fit_transform(x,y)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y,shuffle=True, random_state=333, train_size = 0.8)
model = RandomForestClassifier(random_state=333)
model.fit(x1_train, y1_train)
result1 = model.score(x1_test, y1_test)

print(x.shape,'->',x1.shape)
print('기본acc:',result)
print('LDA_acc:', result1)

# (652, 8) (652,)
# (652, 8) -> (652, 1)
# 기본acc: 0.732824427480916
# LDA_acc: 0.6259541984732825

