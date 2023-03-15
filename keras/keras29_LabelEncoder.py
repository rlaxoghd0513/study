#데이콘 따릉이 문제풀이
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
import pandas as pd



#데이터
path = './_data/dacon_wine/' 


train_csv = pd.read_csv(path + 'train.csv', index_col=0) 


print(train_csv)
print(train_csv.shape)  #(5497. 13)
''''''
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)
print(test_csv.shape)  #(1000,12)
#=======================================================================================

#preprocessing전처리
from sklearn.preprocessing import LabelEncoder, RobustScaler
le = LabelEncoder()
le.fit(train_csv['type'])
aaa = le.transform(train_csv['type'])

print('aaa')
print(type(aaa))  #<class 'numpy.ndarray'>
print(np.unique(aaa, return_counts=True))# return counts 몇개씩 있는지

train_csv['type'] = aaa 
print(train_csv)
print(aaa.shape)  #(5497,)

test_csv['type'] = le.transform(test_csv['type'])


print(le.transform(['red','white']))    #[0 1]  red가 0으로 white가 1로 바뀜
print(le.transform(['white','red']))    #[1 0]
#===================================================================================================

#===================================================================
print(train_csv.columns)

print(train_csv.info()) 


print(train_csv.describe())

print(type(train_csv))

#==================================================
print(train_csv.isnull().sum()) 

train_csv = train_csv.dropna() 
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)


x = train_csv.drop(['count'], axis=1) 

print(x)

y = train_csv['count']
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,shuffle=True, random_state=777, train_size=0.7)
print(x_train.shape, x_test.shape) #(1021,9) (438,9)   (929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(1021,) (438,)     (929,) (399,)

#모델구성
model=Sequential()
model.add(Dense(12, input_dim=9))           
model.add(Dense(16))
model.add(Dense(18))
model.add(Dense(20))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(10))
model.add(Dense(1))

#컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)

#평가 예측
loss=model.evaluate(x_test, y_test)
print('loss=', loss)

y_predict= model.predict(x_test)


r2= r2_score(y_test, y_predict)
print('r2 스코어=', r2)