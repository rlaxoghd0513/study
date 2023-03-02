import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#데이터
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,5,4])

#2. 모델구성
model=Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련  #배치사이즈 디폴트32
model.compile(loss="mse", optimizer="adam")
model.fit(x,y, epochs=10, batch_size=3)

#4. 평가, 예측