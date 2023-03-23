from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D, LSTM,GRU, SimpleRNN, Conv1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

(x_train,y_train), (x_test, y_test)= cifar10.load_data()
print(x_train.shape) # (50000, 32, 32, 3)
print(x_test.shape)  # (10000, 32, 32, 3)

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
print(x_train.shape)  #(50000, 3072)
print(x_test.shape)    #(10000, 3072)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))  #0.0,0.1

print(np.unique(y_train, return_counts=True))   #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 3072,1)
x_test = x_test.reshape(-1, 3072,1)

model = Sequential()
model.add(Conv1D(64,4, input_shape =(3072,1)))
model.add(Conv1D(16,2))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(16))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es= EarlyStopping(monitor = 'acc', mode = 'max', restore_best_weights=True, patience = 30, verbose=1)

model.fit(x_train, y_train, epochs = 100000, batch_size = 128, callbacks = [es],validation_split =0.2 ) 

results = model.evaluate(x_test, y_test)         
print('results:', results)

y_predict = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)
y_predict_acc = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test_acc, y_predict_acc)
print('acc:',acc)