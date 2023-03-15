import numpy as np
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler, MinMaxScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. 데이터
path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
# print(train_csv.shape)        #(5497, 13)


test_csv = pd.read_csv(path+'test.csv', index_col=0)
# print(test_csv.shape)      # (1000, 12)

# print(train_csv.info())  #결측치없음

le = LabelEncoder()
le.fit(train_csv['type'])

# print(type(aaa))  #<class 'numpy.ndarray'>
# print(np.unique(aaa, return_counts=True))# return counts 몇개씩 있는지

train_csv['type'] = le.transform(train_csv['type'])
# print(train_csv['type'])

test_csv['type'] = le.transform(test_csv['type'])



x=train_csv.drop(['quality'], axis=1)
# print(x)    

y=train_csv['quality']

print('y라벨값:',np.unique(y))  # [3 4 5 6 7 8 9]


# =======================================이상치 확인제거  # train_csv['quality'].quantile(0.75) 처럼 특정 열만 적용 가능
q3 = train_csv['quality'].quantile(0.75)    
q1 = train_csv['quality'].quantile(0.25)

iqr = q3 - q1

print(q3)   #6.0
print(q1)    #5.0
print(iqr)    #1.0

upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr

# train_csv_iqr =train_csv['quality']>q3+1.5*iqr
train_csv_iqr=train_csv[(train_csv['quality'] >= lower_bound) & (train_csv['quality'] <= upper_bound)]
train_csv = train_csv_iqr
y= train_csv['quality']
x= train_csv.drop(['quality'], axis=1)
print(x.shape)    # (5340, 12)
print(y.shape)   # (5340,)
print('y라벨값:', np.unique(y))



#============================원핫인코딩================================
ohe= OneHotEncoder()
y = y.values.reshape(-1, 1)
print(y)
# [5]
#  [5]
#  [5]
#  ...
#  [7]
#  [5]
#  [6]]
y = ohe.fit_transform(y).toarray()
print(y)
# [0. 0. 1. ... 0. 0. 0.]
#  [0. 0. 1. ... 0. 0. 0.]
#  [0. 0. 1. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 1. 0. 0.]
#  [0. 0. 1. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]
#=====================================================================



x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1234, shuffle=True, train_size = 0.9,
                                                    stratify=y )
# print(x_train.shape, x_test.shape)      (4397, 11) (1100, 11)
# print(y_train.shape, y_test.shape)      (4397,) (1100,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))   

test_csv = scaler.transform(test_csv) 

# 2 모델구성
input1 = Input(shape=(12, ))
dense1 = Dense(128)(input1)
drop1 = Dropout(0.25)(dense1)
dense2 = Dense(64)(drop1)
dense3 = Dense(32, activation='relu')(dense2)
drop3 = Dropout(0.25)(dense3)
dense4 = Dense(64, activation='relu')(drop3)
drop4 = Dropout(0.25)(dense4)
output1 = Dense(4, activation='softmax')(drop4)
model = Model(inputs=input1, outputs=output1)

#3 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])

import datetime
date = datetime.datetime.now()  #현재시간을 date에 넣어준다
print(date)  #2023-03-14 11:10:57.992357
date = date.strftime("%m%d_%H%M%S") #시간을 문자데이터로 바꾸겠다 그래야 파일명에 넣을 수 있다    %뒤에있는값을 반환해달라
print(date)  #0314_1116

filepath = './_save/MCP/dacon_wine/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', mode='min', patience=300, verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor = 'val_loss', mode= 'auto', save_best_only=True, filepath ="".join([filepath,'DW_',date,'_',filename]), verbose=1)

model.fit(x_train, y_train, batch_size=55, epochs=50,callbacks=[es,mcp], validation_split=0.2)



# 4 평가예측
results = model.evaluate(x_test, y_test)
print('results:', results)

y_predict = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis=1)

y_predict_acc = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test_acc, y_predict_acc)
print('acc:', acc)


# 5 서밋

y_submit = model.predict(test_csv) #submit 제출
# print(y_submit)
y_submit = np.argmax(y_submit, axis=1)
y_submit += 3

submission = pd.read_csv(path+'sample_submission.csv',index_col=0)

                    
# print(submission)
submission['quality'] = y_submit
# print(submission)


submission.to_csv(path_save+ 'wine_'+ date + '.csv')




