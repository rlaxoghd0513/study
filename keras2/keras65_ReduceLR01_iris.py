import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. 데이터
data = load_iris()
x = data.data
y = data.target
# # 3.pandas get_dummies
import pandas as pd
y=pd.get_dummies(y)
# print(y.shape)
# 1-1 train test 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=4))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(3,activation = 'softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

learning_rate = 0.005
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics =['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='acc', patience=20, mode='max', verbose=1)
rlr = ReduceLROnPlateau(monitor='acc', patience=10, mode='auto', verbose=1, factor=0.5)
model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2,
          callbacks=[es, rlr])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)

print("lr:", learning_rate, "loss:", results)