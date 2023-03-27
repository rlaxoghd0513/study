#1 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301,401)])  #예를 들어 삼성, 아모레 주가 데이터 
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]) #온도, 습도, 강수량
print(x1_datasets.shape) #(2,100)
print(x2_datasets.shape) #(3,100)

#결측치를 모델돌려서 예측한 값으로 채워넣을 수도 있고 프리딕트 한번 돌려서 데이터를 부풀려서 한번더 훈련할수도 있다
#머신러닝이 잘맞는 경우도 있다

x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
print(x1) #(100,2)
print(x2) #(100,3)

y = np.array(range(2001, 2101)) #환율

from sklearn.model_selection import train_test_split
# x1_train, x1_test, x2_train, x2_test = train_test_split(x1,x2, train_size = 0.7, random_state = 333)
# y_train, y_test = train_test_split(y, train_size = 0.7, random_state = 333)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1,x2,y, train_size=0.7, random_state=333)  #train_test_split은 3개도 자를수 있다
print(x1_train.shape, x1_test.shape)
print(x2_train.shape, x2_test.shape)
print(y_train.shape, y_test.shape)

#모델구성 
from tensorflow.keras.models import Sequential, Model   #시퀀셜 한개는 되지만 두개 합칠때 안됨 합칠 방법이 없ㅇ므
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1 = Input(shape = (2,))
dense1 = Dense(32 , activation = 'relu', name = 'stock1')(input1) #name 서머리에서 보여주는 이름
dense2 = Dense(64 , activation = 'relu', name = 'stock2')(dense1) #name 서머리에서 보여주는 이름
dense3 = Dense(32 , activation = 'relu', name = 'stock3')(dense2) #name 서머리에서 보여주는 이름
dense4 = Dense(64 , activation = 'relu', name = 'stock4')(dense3) #name 서머리에서 보여주는 이름
output1 = Dense(32 , activation = 'relu', name = 'output1')(dense4) #name 서머리에서 보여주는 이름

#2-2 모델2
input2 = Input(shape = (3,))
dense11 = Dense(16, name = 'weather1')(input2)
dense12 = Dense(32, name = 'weather2')(dense11)
dense13 = Dense(16, name = 'weather3')(dense12)
dense14 = Dense(32, name = 'weather4')(dense13)
output2 = Dense(64, name = 'output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate  #사슬같이엮다 모델을 합친다 소문자는 통상 함수 대문자는 클래스 
merge1 = concatenate([output1 , output2], name = 'mg1') #모델들의 아웃풋들만 연결되면 됨 그게 컨캣한 모델의 인풋이 된다 **두개이상은 리스트** 컨캣한 줄은 연산량이 없다
merge2 = Dense(32, activation = 'relu', name = 'mg2')(merge1) #컨캣한 직후 파라미터 연산량은  (output1노드+ output2노드+ bias)* merge2 노드개수 
merge3 = Dense(16, activation = 'relu', name = 'mg3')(merge2)
last_output = Dense(1, name ='last')(merge3) #output1이나 output2 에서 1개 출력되야하는건 어차피 컨캣해서 last_output까지가 한 모델로 구성되기때문에 1개를 줄 필요가 없다 
                                             #1개주면 오히려 데이터가 소멸되기 때문에  last_output만 맞춰주면 된다

model = Model(inputs = [input1, input2], outputs = last_output)
model.summary()

#컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=32)

loss = model.evaluate([x1_test,x2_test],y_test)
print('loss:', loss)

from sklearn.metrics import mean_squared_error

y_predict = model.predict([x1_test, x2_test])

def RMSE(y_test, y_predict):     #함수정의
    return np.sqrt(mean_squared_error(y_test, y_predict))  #np.sqrt 루트씌우기
rmse = RMSE(y_test, y_predict)    #RMSE 함수 사용
print("RMSE :", rmse)


# loss: 56.03028106689453
# RMSE : 7.4853378597365525

# loss: 18.02634620666504
# RMSE : 4.245744653767112