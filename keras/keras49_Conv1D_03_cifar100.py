import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, GRU, LSTM, Conv1D, Flatten
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

(x_train,y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape)   #(50000, 32, 32, 3)
print(x_test.shape)    #(10000, 32, 32, 3)


x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

print(x_train.shape)  #(50000, 3072)
print(x_test.shape)   #(10000, 3072)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))  #0.0 1.0

print(np.unique(y_train))   #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
#  48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
#  72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
#  96 97 98 99]
print(y_train.shape)  #50000,1

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 96,32)
x_test = x_test.reshape(-1, 96,32)

model = Sequential()
model.add(Conv1D(128,4, input_shape =(96,32)))
model.add(Conv1D(256,3))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(256))
model.add(Dropout(0.4))
model.add(Dense(256))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation = 'ELU'))
model.add(Dense(100, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es= EarlyStopping(monitor = 'acc', mode = 'max', restore_best_weights=True, patience = 100, verbose=1)

model.fit(x_train, y_train, epochs = 10000, batch_size = 64, callbacks = [es],validation_split =0.2 ) 

results = model.evaluate(x_test, y_test)         
print('results:', results)

y_predict = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)
y_predict_acc = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test_acc, y_predict_acc)
print('acc:',acc)