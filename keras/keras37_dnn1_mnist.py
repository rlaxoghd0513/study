from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#[ 실습 ]
#목표: cnn성능보다 좋게만들면 된다
#1 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
print(x_train.shape)  #(60000, 784)
print(x_test.shape)   #(10000, 784)
print(y_train.shape)
print(y_test.shape)
#리쉐이프  

scaler=MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test)) # 0.0 24.0

print(np.unique(y_train, return_counts = True))  #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2 모델구성
model = Sequential()
model.add(Dense(64, input_shape = (784,))) 
# model.add(Dense(64, input_shape = (28*28,))) #이렇게도 된다
model.add(Dense(128))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

es = EarlyStopping(monitor = 'acc', mode = 'max', patience = 50, verbose=1, restore_best_weights=True)
model.compile(loss = 'categorical_crossentropy',optimizer='adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 10000, batch_size = 128, validation_split=0.2 , callbacks = [es])
 
results = model.evaluate(x_test, y_test)
print('results:', results)

y_predict = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)
y_predict_acc = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test_acc, y_predict_acc)
print('acc:', acc)

#acc: 0.9223
