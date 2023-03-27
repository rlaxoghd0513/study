#1 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301,401)])  #예를 들어 삼성, 아모레 주가 데이터 
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]) #온도, 습도, 강수량
x3_datasets = np.array([range(201,301), range(511,611), range(1300, 1400)])
print(x1_datasets.shape) #(2,100)
print(x2_datasets.shape) #(3,100)
print(x3_datasets.shape) #(3, 100)
#결측치를 모델돌려서 예측한 값으로 채워넣을 수도 있고 프리딕트 한번 돌려서 데이터를 부풀려서 한번더 훈련할수도 있다
#머신러닝이 잘맞는 경우도 있다

x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
x3 = x3_datasets.T
print(x1) #(100,2)
print(x2) #(100,3)
print(x3.shape) #(100, 3)

y1 = np.array(range(2001, 2101)) #예:환율
y2 = np.array(range(1001, 1101)) #예:금리


from sklearn.model_selection import train_test_split
# x1_train, x1_test, x2_train, x2_test = train_test_split(x1,x2, train_size = 0.7, random_state = 333)
# y_train, y_test = train_test_split(y, train_size = 0.7, random_state = 333)

# \ 역슬래쉬 '한줄이다' 너무길때 사용
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train,\
y1_test, y2_train, y2_test = train_test_split(x1,x2,x3,y1,y2, train_size=0.7, random_state=333)  #train_test_split은 2개이상도 자를수 있다
print(x1_train.shape, x1_test.shape) #(70, 2) (30, 2)
print(x2_train.shape, x2_test.shape) #(70, 3) (30, 3)
print(x3_train.shape, x3_test.shape) #(70, 3) (30, 3)
print(y1_train.shape, y1_test.shape) #(70,) (30,)

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

#2-3 모델3
input3 = Input(shape = (3,))
dense21 = Dense(16, activation ='relu')(input3)
dense22 = Dense(32)(dense21)
dense23 = Dense(32)(dense22)
dense24 = Dense(32)(dense23)
output3 = Dense(16)(dense24)

#2-4 머지
from tensorflow.keras.layers import concatenate, Concatenate  #사슬같이엮다 모델을 합친다 소문자는 통상 함수 대문자는 클래스 
merge1 = concatenate([output1 , output2, output3], name = 'mg1') #모델들의 아웃풋들만 연결되면 됨 그게 컨캣한 모델의 인풋이 된다 **두개이상은 리스트** 컨캣한 줄은 연산량이 없다
merge2 = Dense(32, activation = 'relu', name = 'mg2')(merge1) #컨캣한 직후 파라미터 연산량은  (output1노드+ output2노드+ bias)* merge2 노드개수 
merge3 = Dense(16, activation = 'relu', name = 'mg3')(merge2)
hidden_output = Dense(1, name ='hidden_output')(merge3)  #얘가 이제 마지막이 아니기때문에 아무 숫자나  넣어도 된다
                                             
#2-5 분기 1
bungi1 = Dense(30, activation='selu', name = 'bg1')(hidden_output)
bungi2 = Dense(15, name = 'bg2')(bungi1)
last_output1 = Dense(1, name = 'last1')(bungi2)

#2-5분기2
last_output2 = Dense(1, activation = 'linear', name = 'last2')(hidden_output)


model = Model(inputs = [input1, input2, input3], outputs =[last_output1, last_output2])
model.summary()

#3 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train,y2_train], epochs=100, batch_size=32)

#4 평가 예측
results = model.evaluate([x1_test,x2_test, x3_test],[y1_test,y2_test])
print('results:', results) #results: [2162.621826171875, 2016.2911376953125, 146.3307342529297] 전체로스 y1로스 y2로스


from sklearn.metrics import mean_squared_error, r2_score

y_predict = np.array(model.predict([x1_test, x2_test, x3_test]))
print(y_predict)
print(y_predict.shape)  #왜 안먹힘?   리스트라서 파이썬 리스트형태라서 len으로 확인 엄밀히 따지면 2,30인데 리스트라서 shape라는 특성 자체가 없다
# (2, 30, 1)                       y_predict에  np.array 씌우면 shape찍을수 있다
# print(len(y_predict), len(y_predict[0]))  #프리딕트 개수, 프리딕트 0번째 열의 개수 


r2_1 = r2_score(y1_test, y_predict[0]) #프리딕트 0번째
r2_2 = r2_score(y2_test, y_predict[1]) #프리딕트 1번째

print('r2스코어 :', (r2_1+r2_2)/2)



# loss1: 127740.484375
# loss2: 413569.1875
# RMSE1 : 357.4080141984253
# RMSE2 : 1.5899990358208536
# r21스코어1 : -215.44245421740857
# r22스코어: 0.9957164129336323