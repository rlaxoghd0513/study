from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Conv1D, Flatten, Conv2D, MaxPooling2D, Reshape, Input
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape)  #(60000, 28, 28) (10000, 28, 28)

x_train = x_train.reshape(-1,28,28,1)/255.  #리쉐잎하면서 스케일링까지 된거 이미지라 255로 나눈다
x_test = x_test.reshape(-1,28,28,1)/255.

print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)



print(np.unique(y_train, return_counts = True))  #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train = x_train.reshape(-1, 28,28)
# x_test = x_test.reshape(-1,28,28)

#2 모델구성
input1 = Input(shape = (28,28,1))
conv1 = Conv2D(64,3,padding = 'same')(input1)
maxp = MaxPooling2D()(conv1)
conv2 = Conv2D(32,3)(maxp)
conv3 = Conv2D(10,3)(conv2)
maxp1 = MaxPooling2D()(conv3)
flat = Flatten()(maxp1)
reshape1 = Reshape(target_shape=(25,10))(flat)
conv4 = Conv1D(10,3,padding='same')(reshape1)
lstm1 = LSTM(784)(conv4)
reshape2 = Reshape(target_shape = (28,28,1))(lstm1)
conv5 = Conv2D(32,3,padding='same')(reshape2)
flat1 = Flatten()(conv5)
output1 = Dense(10, activation = 'softmax')(flat1)
model = Model(inputs =input1, outputs = output1)


es = EarlyStopping(monitor = 'acc', mode = 'max', patience = 50, verbose=1, restore_best_weights=True)
model.compile(loss = 'categorical_crossentropy',optimizer='adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 10000, batch_size = 32, validation_split=0.2 , callbacks = [es])
 
results = model.evaluate(x_test, y_test)
print('results:', results)

y_predict = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)
y_predict_acc = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test_acc, y_predict_acc)
print('acc:', acc)
