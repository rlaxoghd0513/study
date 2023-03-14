from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

#1 데이터
datasets = load_diabetes()
x= datasets.data
y= datasets.target


import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=131, shuffle=True)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))


input1 = Input(shape=(10, ))
dense1 = Dense(12)(input1)
dense2 = Dense(15)(dense1)
# drop1 = Dropout(0.2)(dense2)
dense3 = Dense(17)(dense2)
dense4 = Dense(20)(dense3)
# drop2 = Dropout(0.3)(dense4)
dense5 = Dense(21)(dense4)
dense6 = Dense(15, activation= 'relu')(dense5)
# drop3 = Dropout(0.5)(dense6)
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

y_predict=model.predict(x_test, verbose =0) #x를 넣어서 결과를 보는건 되지만  x중 70퍼는 이미 훈련시킨 데이터

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #훈련시키지 않은 데이터로 평가예측

print('r2스코어 :', r2)

# loss= 2635.789306640625
# r2스코어 : 0.5835873403907975

# r2스코어 : 0.3520995804014333
# loss= 4101.04931640625