import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#1 . 데이터
datasets = np.array([1,2,3,4,5,6,7,8,10])
#y=?
x = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8],[6,7,8,9]]) 
y = np.array([5,6,7,8,9,10])

print(x.shape, y.shape) #(6, 4)(6,)  #rnn은 통상 3차원데이터 훈련
# x의 shape = (행, 열, 몇개씩 훈련하는지)

x = x.reshape(6,2,2) #[[[1],[2],[3]],[[2],[3],[4]],......]
print(x.shape)  #(6, 4, 1)
print(x)

#2 모델구성2
model = Sequential()
model.add(SimpleRNN(10, input_shape = (2,2)))  #(5,1)이만큼씩 훈련시키겠다 32는 아웃풋 노드갯수
model.add(Dense(7))
model.add(Dense(1))
model.summary()

