from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

#1 데이터
datasets = load_diabetes()
x= datasets.data
y= datasets.target


import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=131, shuffle=True)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

# model=Sequential()
# model.add(Dense(12, input_dim=10))
# model.add(Dense(15))
# model.add(Dense(17))
# model.add(Dense(20))
# model.add(Dense(21))
# model.add(Dense(15))
# model.add(Dense(13))
# model.add(Dense(8))
# model.add(Dense(1))

input1 = Input(shape=(10, ))
dense1 = Dense(12)(input1)
dense2 = Dense(15)(dense1)
dense3 = Dense(17)(dense2)
dense4 = Dense(20)(dense3)
dense5 = Dense(21)(dense4)
dense6 = Dense(15, activation= 'relu')(dense5)
dense7 = Dense(13, activation= 'relu')(dense6)
dense8 = Dense(8)(dense7)
output1 = Dense(1)(dense8)
model = Model(inputs = input1, outputs = output1 )

#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
es=EarlyStopping(monitor='val_loss', patience = 20, mode='min'
              ,verbose=1
              ,restore_best_weights=True) 
hist = model.fit(x_train, y_train, epochs=100 , batch_size=4, validation_split=0.2, callbacks=[es])
print(hist.history['val_loss'])

#평가 예측
loss=model.evaluate(x_test, y_test)
print('loss=', loss)

y_predict=model.predict(x_test) #x를 넣어서 결과를 보는건 되지만  x중 70퍼는 이미 훈련시킨 데이터

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #훈련시키지 않은 데이터로 평가예측

print('r2스코어 :', r2)

#sequential loss 2642.20 r2 0.58
#model      loss 2707.49 r2 0.57