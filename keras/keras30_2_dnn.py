from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()                          #(N,3)
model.add(Dense(10, input_shape=(3,))) #(batch_size, input_dim) input_dim만 명시중    #3차원의 벡터의 배치
model.add(Dense(units = 15)) #출력 (batch_size, units)
model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 10)                40
# _________________________________________________________________
# dense_1 (Dense)              (None, 15)                165
# =================================================================
# Total params: 205
# Trainable params: 205
# Non-trainable params: 0
# _________________________________________________________________