import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
#Rnn
#시계열 데이터는 과거의 데이터들이 현재에 영향을 끼친다
#첫번째에서 뽑아낸 y'가 두번째에 더해진다 근데 그럼 너무 큰값이 나오니까 activation 디폴트 tan해서 -1에서 1사이의 값으로 준다
#지도학습  y의 값을 알때 
#시계열데이터에는 y가 없다 내가 정한다

#1 . 데이터
datasets = np.array([1,2,3,4,5,6,7,8,10])
#y=?
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])   #8910넣으면 그다음을 모른다
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) #(7, 3)(7,)  #rnn은 통상 3차원데이터 훈련
# x의 shape = (행, 열, 몇개씩 훈련하는지)

x = x.reshape(7,3,1) #[[[1],[2],[3]],[[2],[3],[4]],......]
print(x.shape)  #(7, 3, 1)

#2 모델구성
model = Sequential()
model.add(SimpleRNN(64, input_shape = (3,1)))  #(3,1)이만큼씩 훈련시키겠다 32는 아웃풋 노드갯수
model.add(Dense(32, activation='relu'))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

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