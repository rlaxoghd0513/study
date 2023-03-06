from sklearn.datasets import load_boston

#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.7, random_state=9010, shuffle=True)


# print(x)
# print(y)
# print(datasets)
# print(datasets.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

# print(datasets.DESCR) #instace 예시 attribute 특성

# print(x.shape, y.shape) #(504, 13) (504, )



#모델구성
model=Sequential()
model.add(Dense(14, input_dim=13))
model.add(Dense(11))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mse', optimizer='adam') #rmse mse제곱이라 루트 씌운거
model.fit(x_train, y_train, epochs=100, batch_size=8, verbose='auto')

# ***keras.io 참고***
# verbose 말이 겁나 많은  
# 훈련과정 자세히 보일지 말지 
# 0 끄기 1켜기 2프로그레스바 끄기 0,1,2,auto빼고 나머지는 에포만
# auto 다보여줌

#평가예측
loss=model.evaluate(x_test, y_test, verbose='auto')
print('loss=', loss)

y_predict=model.predict(x_test) #x를 넣어서 결과를 보는건 되지만  x중 70퍼는 이미 훈련시킨 데이터

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #훈련시키지 않은 데이터로 평가예측

print('r2스코어 :', r2)
