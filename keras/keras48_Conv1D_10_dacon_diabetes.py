import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input,Dropout, Conv2D, Flatten, GRU, SimpleRNN, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import RobustScaler,StandardScaler, MaxAbsScaler, MinMaxScaler
#데이터
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

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=88484, shuffle=True, train_size=0.9)
# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
#(521, 8) (131, 8)
#(521,) (131,)
scaler= RobustScaler()  #분류에 더 유용한 스케일러  standard 이상치에 매우 민감감 robust
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))
print(np.unique(y_train))

test_csv = scaler.transform(test_csv)  #train_csv도 스케일링 했으니까 제출할 test_csv 도 마찬가지로 스케일링 해줘야한다

print(x_train.shape)  #(586, 8)
print(x_test.shape)    #(66, 8)
x_train = x_train.reshape(586,4,2)
x_test = x_test.reshape(66,4,2)

input1 = Input(shape = (4,2))
conv1 = Conv1D(16,3,padding='same')(input1)
conv2 = Conv1D(32,2)(conv1)
flat = Flatten()(conv2)
dense1 = Dense(16)(flat)
dense2 = Dense(32)(dense1)
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(16, activation='relu')(drop1)
output1 = Dense(1, activation='sigmoid')(dense3)
model = Model(inputs =input1, outputs = output1)

#컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
import datetime
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M") 
 

filepath = './_save/MCP/dacon_diabetes/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

es=EarlyStopping(monitor = 'accuracy', patience=150, mode='max', verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor = 'accuracy', mode= 'max', save_best_only=True, verbose=1, filepath ="".join([filepath,'dia_',date,'_',filename]))

model.fit(x_train,y_train, epochs=10000, batch_size=1, validation_split=0.2, callbacks=[es,mcp])


#평가 에측
results = model.evaluate(x_test, y_test)
print('results:', results)

y_predict = np.round(model.predict(x_test))

acc=accuracy_score(y_test, y_predict)
print('acc:',acc)
