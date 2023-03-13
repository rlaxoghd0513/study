#input_dim 행과 열만 있는 2차원 데이터셋만 했음
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model  #함수형모델은 그냥 Model
from tensorflow.python.keras.layers import Dense, Input    #함수형에서는 따로 명시해야됨
import numpy as np
from sklearn.preprocessing import MinMaxScaler #값을 0에서 1사이로 바꾸지만 standardScaler는 평균점을 중심으로 모아준다
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MaxAbsScaler  
from sklearn.preprocessing import RobustScaler  

#1 데이터
datasets = load_boston()
x= datasets.data    
y= datasets.target

print(type(x)) #<class 'numpy.ndarray'>
print(x)



# print(np.min(x), np.max(x))  #0.0 711.0
# scaler = MinMaxScaler()
# scaler.fit(x)  #이 비율로 변환할 준비를 해라
# x = scaler.transform(x) # 실제로 변환해라  fit과 transform 둘다 실행해야한다 #0.0 1.0
# print(np.min(x), np.max(x)) 

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=300)

scaler = MinMaxScaler()  #scaler = StandardScaler()    뭐쓸지만 정하면 됨
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) #fit과 transform 두줄을 한줄로 하는거---->> x_train = scaler.fit_transform(x_train) , x_test와 test_csv 여전히 transform만 하면 된다
x_train = scaler.fit_transform(x_train)


# x_test는 x_train의 변한 범위에 맞춰서 하기 때문에 fit 할 필요가 없다 scaler.fit(x_train)그대로 써야됨
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test)) 



#2 모델구성
# model=Sequential()
# model.add(Dense(1, input_dim=13))
# model.add(Dense(30, input_shape=(13, )))    #함수형에서는 input_layer랑 첫번재 히든레이어랑 같은 줄에 쓰면 안되고 따로 명시해야한다
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))

input1 = Input(shape=(13,))
dense1 = Dense(30)(input1) #sequential은 순서대로 함수형은 모델을 구성하고 시작이 어딘지 끝이 어딘지 마지막에 정해준다
dense2 = Dense(20)(dense1) #이 레이어가 어디서 왔는지 꽁다리에 
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3) 
model = Model(inputs = input1, outputs = output1)   #sequential은 순서대로 함수형은 모델을 구성하고 시작이 어딘지 끝이 어딘지 마지막에 정해준다

# model.save('./_save/keras26_3_save_model.h5') 
model.save_weights('./_save/keras26_5_save_weights1.h5')



#컴파일 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)

# model.save('./_save/keras26_3_save_model.h5') 
model.save_weights('./_save/keras26_5_save_weights2.h5')

#평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)