from tensorflow.keras.datasets import imdb
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print(x_train)
print(y_train)
print(x_train.shape, x_test.shape)      # (25000,) (25000,)
print(np.unique(y_train, return_counts=True))             # [0 1]
print(pd.value_counts(y_train))
# 1    12500
# 0    12500

print(max(len(i) for i in x_train))                    # 2494
print(sum(map(len, x_train))/len(x_train))             # 238.71364

pad_x_train = pad_sequences(x_train, maxlen=100, padding='pre', truncating='pre')
pad_x_test = pad_sequences(x_test, maxlen=100, padding='pre', truncating='pre')
pad_x_train = pad_x_train.reshape(pad_x_train.shape[0], pad_x_train.shape[1], 1)
pad_x_test = pad_x_test.reshape(pad_x_test.shape[0], pad_x_test.shape[1], 1)
# 2. 모델
model = Sequential()
model.add(Embedding(10000, 32, input_shape=(100,)))
model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(pad_x_train, y_train, batch_size=64, epochs=100, validation_split=0.2)

# 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]



print('acc : ', acc)