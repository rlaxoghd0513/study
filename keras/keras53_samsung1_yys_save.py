#삼성전자와 현대자동차 주가로 삼성전자 주가 맞추기

#각각 데이터에서 컬림 7개 이상 추출( 그중 거래량은 반드시 들어갈 것)
#timesteps와 feature는 알아서 잘라라

#제공된 데이터 외 추가 데이터 사용금지

#1 삼성전자 28일 종가 맞추기 (점수배점 .3)
#2 삼성전자 29일 아침 시가 맞추기 (점수배점 .7)

# 27일 23시 59분
#메일 제목 : 윤영선 [삼성1차] 60.350.07원
#              윤영선 [삼성 2차] 60.350.07원

# 첨부파일 : keras53_samsung2_kth_submit.py  가중치 불러오고 데이터 불러오고 
#           keras_samsung4_kth_submit.py

# 가중치 _save/samsung/keras53_samsung2_kth.h5 hdf5
#       _save/samsung/keras53_samsung4_kth.h5 hdf5
import numpy as np
import pandas as pd

path = './_data/시험/'
path_save = './_save/시험/'

datasets1 = pd.read_csv(path + '삼성전자 주가2.csv', index_col = 0, header=0)
datasets2 = pd.read_csv(path + '현대자동차.csv', index_col =0, header=0)

datasets1 = datasets1.drop(['전일비'], axis=1)
datasets2 = datasets2.drop(['전일비'], axis=1)
datasets1 = datasets1.drop(['Unnamed: 6'], axis=1)
datasets2 = datasets2.drop(['Unnamed: 6'],axis=1)

print(type(datasets1)) #<class 'pandas.core.frame.DataFrame'>
print(type(datasets2)) #<class 'pandas.core.frame.DataFrame'>
print(datasets1.info())
print(datasets2.info())
##########################################따움표제거######################################################
(datasets1['종가']).apply(lambda x: int(x))

    
print(datasets1.info())
    


#########################################################################################################
datasets1 = datasets1.sort_values('일자', ascending=True)
datasets2 = datasets2.sort_values('일자', ascending=True)

datasets1 = datasets1.iloc[120:]

##########################################################################################################



x1 = datasets1.drop(['종가'],axis=1)
y = datasets1['종가']
x2 = datasets2






print(type(x1))
print(type(x2))
print(type(y))
print(x1.info())
print(x2.info())
print(y.info())
###########################################################

##########################################################
timesteps = 10
def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps):  #스플릿한 x_daata중 마지막꺼 못쓰니까 +1없애준다  
        subset = dataset[i : (i+timesteps)] 
        aaa.append(subset)      
    return np.array(aaa)

x1 = split_x(x1, timesteps)
# print(x1)
print(x1.shape) #(3130, 10, 14)

x2 = split_x(x2, timesteps)
# print(x2)
print(x2.shape) #(3130, 10, 15)

y = y[timesteps:]
print(y.shape) #(3130,)
##############################



from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1,x2,y, random_state = 333, train_size = 0.8)
print(x1_train.shape, x1_test.shape ) #(2504, 10, 14) (626, 10, 14)
print(x2_train.shape, x2_test.shape) #(2504, 10, 15) (626, 10, 15)
print(y_train.shape , y_test.shape) #(2504,) (626,)

x1_train = x1_train.reshape(2504,140)
x1_test = x1_test.reshape(626,140)
x2_train = x2_train.reshape(2504,150)
x2_test = x2_test.reshape(626,150)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)

x2_train = scaler.fit_transform(x2_train)
x2_test = scaler.transform(x2_test)

print(np.min(x1_train), np.max(x1_train))
print(np.min(x1_test), np.max(x1_test)) 
print(np.min(x2_train), np.max(x2_train))
print(np.min(x2_test), np.max(x2_test))

x1_train = x1_train.reshape(2504,10,14)
x1_test = x1_test.reshape(626,10,14)
x2_train = x2_train.reshape(2504,10,15)
x2_test = x2_test.reshape(626,10,15)




#모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input, Bidirectional, Flatten, Dropout, Dense
#모델1
input1 = Input(shape = (10,14))
lstm1 = LSTM(32, return_sequences=True)(input1)
lstm2 = LSTM(64, return_sequences=True)(lstm1)
lstm3 = LSTM(32)(lstm2)
dense1 = Dense(64)(lstm3)
dense2 = Dense(32)(dense1)
output1 = Dense(16)(dense2)
#모델2
input2 = Input(shape = (10,15))
lstm11 = LSTM(32, return_sequences=True)(input2)
lstm12 = LSTM(64)(lstm11)
dense11 = Dense(32)(lstm12)
dense12 = Dense(64)(dense11)
output2 = Dense(16)(dense12)
#머지
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(32)(merge1)
merge3 = Dense(16)(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs = [input1,input2], outputs = last_output)
model.summary()

#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train],y_train, epochs=1, batch_size=32)

#평가 예측
results = model.evaluate([x1_test, x2_test], y_test)
print('results', results)

from sklearn.metrics import r2_score
y_predict = model.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_predict)
print('r2스코어:', r2)