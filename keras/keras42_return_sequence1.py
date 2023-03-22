import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU


#LSTM 구조 3개의 게이트와 한개의 스테이트 
# (forgettable gate input gate output gate)   (cell state)
#구조 꼭 기억해야함
#파라미터 연산값 simplernn 4배

#1 . 데이터
datasets = np.array([1,2,3,4,5,6,7,8,10])
#y=?
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]
             ,[5,6,7],[6,7,8],[7,8,9],[8,9,10]
             ,[9,10,11],[10,11,12]
             ,[20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50,60,70])

print(x.shape, y.shape) #(13,3)(13,)
# x의 shape = (행, 열, 몇개씩 훈련하는지)

x = x.reshape(13,3,1) #[[[1],[2],[3]],[[2],[3],[4]],......]
print(x.shape)  #(13, 3, 1)

#2 모델구성
model = Sequential()
# model.add(LSTM(64, input_shape = (3,1)))  #(3,1)이만큼씩 훈련시키겠다 32는 아웃풋 노드갯수
model.add(LSTM(10, input_shape=(3,1), return_sequences=True))    #return_sequences=True 결과값을 2차원이 아닌 3차원으로 내겠다 그래서 다음 LSTM이나 GRU와 연결가능
model.add(LSTM(11))
model.add(GRU())
model.add(Dense(1))
model.summary()

'''
#3 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x,y, epochs = 150, batch_size = 1)

#4 평가 예측
loss = model.evaluate(x,y)
x_predict = np.array([8,9,10]).reshape(1,3,1) #[[[8],[9],[10]]]
print(x_predict.shape) #(1,3,1)

results = model.predict(x_predict)
print('loss:', loss)
print('[8,9,10]의 결과:', results)
'''