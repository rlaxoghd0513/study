import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input,LeakyReLU
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.callbacks import EarlyStopping

path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)



print(train_csv.isnull().sum()) 
x = train_csv.drop(['quality','type'], axis= 1)
y = train_csv['quality']
test_csv = test_csv.drop(['type'],axis = 1)

ohe = OneHotEncoder()
print(type(y))   
print(y) 

y = train_csv['quality'].values.reshape(-1, 1)
print(y)

y = ohe.fit_transform(y).toarray()
print(y)



print(y.shape)   



x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, random_state= 1742, train_size= 0.7, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2.모델구성

input1 = Input(shape = (11,))
dense1 = Dense(128)(input1)
drop1 = Dropout(0.25)(dense1)
dense2 = Dense(64, activation= LeakyReLU(0.85))(drop1)
dense3 = Dense(32)(dense2)
drop2 = Dropout(0.25)(dense3)
dense4 = Dense(64, activation= LeakyReLU(0.85))(drop2)
drop3 = Dropout(0.25)(dense4)
dense5 = Dense(64, activation= LeakyReLU(0.85))(drop3)
dense6 = Dense(64, activation= LeakyReLU(0.85))(dense5)
drop4 = Dropout(0.25)(dense5)
output1 = Dense(7, activation= 'softmax')(drop4)
model = Model(inputs = input1, outputs = output1)



es = EarlyStopping(monitor= 'val_acc', patience= 300, mode = 'max',
                   restore_best_weights= True,
                   verbose= 1)

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc'])

model.fit(x_train, y_train, epochs = 5000, 
          batch_size = 55, verbose = 1,
          validation_split= 0.2,
          callbacks = [es])


results = model.evaluate(x_test,y_test)
print('results :', results)

y_predict = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis =-1)
y_predict = np.argmax(y_predict, axis =-1)


acc = accuracy_score(y_test_acc, y_predict)
print('Accuary score : ', acc)

#minmax
#activation elu
#opti adamax adam
#batch 32
#epoch 100
