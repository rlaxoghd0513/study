from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, LSTM, SimpleRNN, Conv1D, Flatten
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

(x_train,y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)    #(60000, 28, 28)
print(x_test.shape)    #(10000, 28, 28)

x_train  = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000, 28*28)
print(x_train.shape)
print(x_test.shape)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 28,28)
x_test = x_test.reshape(-1, 28,28)

print(np.unique(y_train))     # [0 1 2 3 4 5 6 7 8 9]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv1D(32,4 ,input_shape = (28,28))) 
# model.add(Dense(64, input_shape = (28*28,))) #이렇게도 된다
model.add(Conv1D(64,3))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dropout(0.25))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer='adam', metrics = ['acc'])
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 100, verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs = 10000, batch_size = 32 ,callbacks = [es], validation_split = 0.2)
 
results = model.evaluate(x_test, y_test)
print('results:', results)

y_predict = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)
y_predict_acc = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test_acc, y_predict_acc)
print('acc:', acc)