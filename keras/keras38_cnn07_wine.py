#사이킷런 load_wine
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)#(178, 13),(178,)

# print(x)
# print(y)
print('y의 라벨값:', np.unique(y)) #y의 라벨값 [012]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) #(178, 3)

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=333, train_size=0.8, stratify=y )

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

print(x_train.shape) #(142, 13))
print(x_test.shape) #(36, 13)

x_train = x_train.reshape(142,13,1,1)
x_test = x_test.reshape(36,13,1,1)



input1 = Input(shape=(13,1,1))
conv1 = Conv2D(16, (2,1), padding='same')(input1)
conv2 = Conv2D(32, (2,1), padding='same')(conv1)
conv3 = Conv2D(16, (2,1), padding='same')(conv2)
flat = Flatten()(conv3)
dense1 = Dense(16)(flat)
dense2 = Dense(32)(dense1)
dense3 = Dense(16)(dense2)
drop1 = Dropout(0.2)(dense3)
dense4 = Dense(16, activation='relu')(drop1)
output1 = Dense(3, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)



model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])  
es = EarlyStopping(monitor = 'acc', mode = 'max', patience=20,verbose = 1, restore_best_weights=True )
model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split = 0.2, verbose=1, callbacks= [es])
print(y_train.shape) #(142,3)
print(y_test.shape) #(36, 3)


results = model.evaluate(x_test, y_test)
print('results:', results)
y_predict = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis=1)
y_predict_acc = np.argmax(y_predict, axis = 1) 
print(y_predict)
acc=accuracy_score(y_test_acc, y_predict_acc)
print('acc:',acc)
