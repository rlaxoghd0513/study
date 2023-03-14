import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터
path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
# print(train_csv.shape)        #(5497, 13)


test_csv = pd.read_csv(path+'test.csv', index_col=0)
# print(test_csv.shape)      # (1000, 12)

# print(train_csv.info())  #결측치없음
test_csv = test_csv.drop(['type'], axis=1)


x=train_csv.drop(['quality','type'], axis=1)
# print(x)    #[5497 rows x 12 columns]

y=train_csv['quality']

print('y라벨값:',np.unique(y))  # [3 4 5 6 7 8 9]

y=to_categorical(y)
# print(y.shape)  #(5497, 10)
# y= np.delete(y,[0,1,2],axis=1)
# print(y.shape)  #(5497, 7)
# 2. sklearn

# ohe = OneHotEncoder()
# y = y.reshape(-1,1)
# y = ohe.fit_transform(y).toarray()
# print(y.shape) # (581012,7)
# print(type(y))



x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=123, shuffle=True, train_size = 0.8)
# print(x_train.shape, x_test.shape)      (4397, 11) (1100, 11)
# print(y_train.shape, y_test.shape)      (4397,) (1100,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))   

test_csv = scaler.transform(test_csv) 

#2 모델구성
input1 = Input(shape=(11, ))
dense1 = Dense(15)(input1)
dense1 = Dense(14)(input1)
dense2 = Dense(13)(dense1)
dense3 = Dense(11, activation='relu')(dense2)
dense4 = Dense(8, activation='relu')(dense3)
output1 = Dense(10, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)

#3 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])

import datetime
date = datetime.datetime.now()  #현재시간을 date에 넣어준다
print(date)  #2023-03-14 11:10:57.992357
date = date.strftime("%m%d_%H%M") #시간을 문자데이터로 바꾸겠다 그래야 파일명에 넣을 수 있다    %뒤에있는값을 반환해달라
print(date)  #0314_1116

filepath = './_save/MCP/dacon_wine/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor = 'val_loss', mode= 'auto', save_best_only=True, filepath ="".join([filepath,'DW_',date,'_',filename]))

model.fit(x_train, y_train, batch_size=32, epochs=1000,callbacks=[es,mcp], validation_split=0.2)

# 4 평가예측
results = model.evaluate(x_test, y_test)
print('results:', results)

y_predict = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis=1)

y_predict_acc = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test_acc, y_predict_acc)
print('acc:', acc)

# 5 서밋

y_submit = np.argmax(model.predict(test_csv), axis=1) #submit 제출
# print(y_submit)

submission = pd.read_csv(path+'sample_submission.csv',index_col=0)
                    
# print(submission)
submission['quality'] = y_submit
# print(submission)

submission.to_csv(path_save+'submit_0314_1658.csv')




