#삼성전자와 현대자동차 주가로 삼성전자 주가 맞추기

#각각 데이터에서 컬림 7개 이상 추출( 그중 거래량은 반드시 들어갈 것)
#timesteps와 feature는 알아서 잘라라

#제공된 데이터 외 추가 데이터 사용금지

#1 삼성전자 29일 종가 맞추기 (점수배점 .3)
#2 현대 30일 아침 시가 맞추기 (점수배점 .7)

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

datasets1 = pd.read_csv(path + '삼성전자 주가3.csv', index_col = 0, encoding = 'utf-8', thousands = ',')
datasets2 = pd.read_csv(path + '현대자동차2.csv', index_col =0, encoding = 'utf-8', thousands = ',')

print(datasets1.head)

datasets1 = datasets1.drop(['전일비'], axis=1)
datasets2 = datasets2.drop(['전일비'], axis=1)
datasets1 = datasets1.drop(['Unnamed: 6'], axis=1)
datasets2 = datasets2.drop(['Unnamed: 6'],axis=1)
datasets1 = datasets1.drop(['외인(수량)'], axis=1)
datasets2 = datasets2.drop(['외인(수량)'], axis=1)
datasets1 = datasets1.drop(['외국계'], axis=1)
datasets2 = datasets2.drop(['외국계'], axis=1)
datasets1 = datasets1.drop(['프로그램'], axis=1)
datasets2 = datasets2.drop(['프로그램'], axis=1)
datasets1 = datasets1.drop(['기관'], axis=1)
datasets2 = datasets2.drop(['기관'], axis=1)

# print(type(datasets1)) #<class 'pandas.core.frame.DataFrame'>
# print(type(datasets2)) #<class 'pandas.core.frame.DataFrame'>
print(datasets1.info())
print(datasets2.info())

datasets1 = datasets1.iloc[:1207]
datasets2 = datasets2.iloc[:1207]
###########################################################
datasets1 = datasets1.sort_values('일자', ascending=True)
datasets2 = datasets2.sort_values('일자', ascending=True)

print(datasets1.head)

x1 = datasets1.drop(['종가'], axis=1)
x2 = datasets2.drop(['종가'], axis=1)
y = datasets1['종가']

# print(x1.isnull().sum())
# print(x2.isnull().sum())



# import matplotlib.pyplot as plt
# # plt.plot(x1['거래량']) #거래량 robust
# # plt.plot(x1['등락률']) #등락률 robust
# # plt.plot(x1['금액(백만)']) #금액 백만 robust
# # plt.plot(x1['개인']) #개인 robust
# # plt.plot(x1['기관']) #기관 robust
# plt.plot(x1['외인비']) #외인(수량) robust 외국계 robust 프로그램
                  
# plt.show()
##########################################################
timesteps = 10
def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps):  #스플릿한 x_daata중 마지막꺼 못쓰니까 +1없애준다  
        subset = dataset[i : (i+timesteps)] 
        aaa.append(subset)      
    return np.array(aaa)

x1 = split_x(x1, timesteps)

x2 = split_x(x2, timesteps)

y = y[timesteps:]

#########################################################


from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1,x2,y, random_state = 158796, train_size = 0.9)
print(x1_train.shape, x1_test.shape) #(1077, 10, 9) (120, 10, 9)
print(x2_train.shape, x2_test.shape) #(1077, 10, 9) (120, 10, 9)

x1_train = x1_train.reshape(1077,90)
x1_test = x1_test.reshape(120,90)
x2_train = x2_train.reshape(1077,90)
x2_test = x2_test.reshape(120,90)

from sklearn.preprocessing import MinMaxScaler, RobustScaler
scaler = MinMaxScaler()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)

x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)


x1_train = x1_train.reshape(1077,10,9)
x1_test = x1_test.reshape(120,10,9)
x2_train = x2_train.reshape(1077,10,9)
x2_test = x2_test.reshape(120,10,9)



#모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input, Bidirectional, Flatten, Dropout, Dense
#모델1
input1 = Input(shape = (10,9))
lstm1  = Bidirectional(LSTM(32, return_sequences=True))(input1)
lstm2 = Bidirectional(LSTM(64, return_sequences=True))(lstm1)
lstm3 = LSTM(32)(lstm2)
dense1 = Dense(64)(lstm3)
dense2 = Dense(32)(dense1)
dense3 = Dense(16)(dense2)
output1 = Dense(8)(dense3)
#모델2
input2 = Input(shape = (10,9))
lstm11 = LSTM(32, return_sequences=True)(input2)
lstm12 = Bidirectional(LSTM(64, return_sequences=True))(lstm11)
lstm13 = Bidirectional(LSTM(32))(lstm12)
dense11 = Dense(64)(lstm13)
dense14 = Dense(32)(dense11)
output2 = Dense(16)(dense14)
#머지
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(32)(merge1)
merge3 = Dense(64)(merge2)
merge4 = Dense(16)(merge3)
last_output = Dense(1)(merge4)

model = Model(inputs = [input1,input2], outputs = last_output)
# model.summary()

#컴파일 훈련
model.compile(loss='mse', optimizer='adamax')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath = './_save/MCP/시험/'
filename = '{epoch:04d}-{val_loss:.2f}.hdf5'

import datetime
date = datetime.datetime.now() 
date = date.strftime("%m%d_%H%M")

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 30, restore_best_weights=True)
mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'min',verbose=1, save_best_only=True, filepath ="".join([filepath,'시험_',date,'_',filename]))
model.fit([x1_train, x2_train],y_train, epochs=10000, batch_size=32, validation_split=0.2, callbacks=[es,mcp])

#평가 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss:',loss)

y_predict =np.round(model.predict([x1_test, x2_test]),2)

print(y_predict[-1:])


