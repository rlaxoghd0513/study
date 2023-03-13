#input_dim 행과 열만 있는 2차원 데이터셋만 했음
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model #모델불러오기
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




#모델
model = load_model('./_save/keras26_3_save_model.h5') # 모델 불러오기 save할때 compile fit 뒤에서 save하면 모델과 가중치까지 저장 , compile fit 앞에서 save하면 모델만 저장 
model.summary()                                                                #가중치까지 저장된 모델을 불러와서 다시 compile fit시키면 다시 훈련시켜서 불러온 가중치 날라감


#컴파일 훈련
# model.compile(loss='mse', optimizer= 'adam')
# model.fit(x_train, y_train, epochs=100, batch_size=32)

#평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)