import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, GRU, SimpleRNN, LSTM, Conv1D
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping

#1데이터
datasets = load_iris()
print(datasets.DESCR) 


x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)

print(x)
print(y)
print('y의 라벨값:', np.unique(y))   #y의 라벨값 [0,1,2]


from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) #(150, 3)


x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, 
                                                    random_state=333,
                                                    train_size=0.8,
                                                    stratify=y 
                                                    )
print(y_train)
print(np.unique(y_train, return_counts=True)) 

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

print(x_train.shape)  #(120, 4)
print(x_test.shape)  #(30, 4)

x_train = x_train.reshape(120,2,2)
x_test = x_test.reshape(30,2,2)


input1 = Input(shape=(2,2))
conv1 = Conv1D(16,2, padding='same')(input1)
conv2 = Conv1D(32,2, padding='same')(conv1)
conv3 = Conv1D(16)(conv2)
flat = Flatten()(conv3)
dense1 = Dense(16)(flat)
dense2 = Dense(8)(dense1)
dense3 = Dense(16, activation ='relu')(dense2)
drop1 = Dropout(0.3)(dense3)
output1 = Dense(3, activation='softmax')(drop1)
model = Model(inputs= input1, outputs=output1)

#3 컴파일 훈련
es = EarlyStopping(monitor='acc', patience = 20, mode='max'
              ,verbose=1
              ,restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])  #다중분류에서 loss는 
model.fit(x_train, y_train, epochs=1, batch_size = 1, validation_split = 0.2, verbose=1)

#4. 평가 예측
results = model.evaluate(x_test, y_test) 
print(results)    
print('loss:', results[0])
print('acc:',results[1])

y_predict = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis=1)  
y_pred = np.argmax(y_predict, axis = 1) 
acc = accuracy_score(y_test_acc, y_pred) 

print('acc:', acc)
