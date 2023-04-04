from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd

#1 데이터
(x_train, y_train), (x_test,y_test) = reuters.load_data(
    num_words=10000, test_split=0.2   #num_words 상위키워드 몇개만 나와라
)

print(x_train)
print(y_train)
print(x_train.shape, y_train.shape) #(8982,) (8982,)
print(x_test.shape, y_test.shape) #(2246,) (2246,)

print(len(x_train[0]), len(x_train[1])) #87 56 넘파이 형태로 리스트

print(np.unique(y_train))
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

print(type(x_train), type(y_train)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0])) #<class 'list'> 

print("뉴스기사의 최대길이:", max(len(i) for i in x_train)) #x_train중에서 제일 긴값  2376
print("뉴스기사의 평균길이:", sum(map(len, x_train))/len(x_train)) #  145.5398574927633

#전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, padding = 'pre', maxlen = 100, 
                        truncating='pre') #padding 어딜채울거냐 truncating 어딜버릴거냐

print(x_train.shape) #(8982, 100)

x_test = pad_sequences(x_test, padding='pre', maxlen = 100,
                       truncating='pre')

print(x_test.shape) #(2246, 100)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM

pad_x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
pad_x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

model = Sequential()
model.add(Embedding(10000,32, input_length=100))
model.add(LSTM(32))
model.add(Dense(32))
model.add(Dense(46, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
model.fit(pad_x_train, y_train, epochs = 1)

acc = model.evaluate(pad_x_test, y_test)[1]
print(acc)
