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

x1 = datasets1.drop(['시가'], axis=1)
x2 = datasets2.drop(['시가'], axis=1)
y = datasets2['시가']

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
    for i in range(len(dataset) - timesteps-1):  #스플릿한 x_daata중 마지막꺼 못쓰니까 +1없애준다  
        subset = dataset[i : (i+timesteps)] 
        aaa.append(subset)      
    return np.array(aaa)

x1 = split_x(x1, timesteps)

x2 = split_x(x2, timesteps)

y = y[timesteps+1:]

#########################################################


from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1,x2,y, random_state = 158796, train_size = 0.8)
print(x1_train.shape, x1_test.shape) #(956, 10, 9) (240, 10, 9)
print(x2_train.shape, x2_test.shape) #(956, 10, 9) (240, 10, 9)

x1_train = x1_train.reshape(956,90)
x1_test = x1_test.reshape(240,90)
x2_train = x2_train.reshape(956,90)
x2_test = x2_test.reshape(240,90)

from sklearn.preprocessing import MinMaxScaler, RobustScaler
scaler = MinMaxScaler()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)

x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)


x1_train = x1_train.reshape(956,10,9)
x1_test = x1_test.reshape(240,10,9)
x2_train = x2_train.reshape(956,10,9)
x2_test = x2_test.reshape(240,10,9)



#모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input, Bidirectional, Flatten, Dropout, Dense
#모델1
input1 = Input(shape = (10,9))
lstm1  = Bidirectional(LSTM(32, return_sequences=True))(input1)
lstm2 = Bidirectional(LSTM(64, return_sequences=True))(lstm1)
lstm3 = LSTM(128)(lstm2)
dense1 = Dense(64)(lstm3)
dense2 = Dense(32)(dense1)
dense3 = Dense(64)(dense2)
output1 = Dense(32)(dense3)
#모델2
input2 = Input(shape = (10,9))
lstm11 = LSTM(64, return_sequences=True)(input2)
lstm12 = Bidirectional(LSTM(128, return_sequences=True))(lstm11)
lstm13 = Bidirectional(LSTM(64))(lstm12)
dense11 = Dense(32)(lstm13)
dense14 = Dense(64)(dense11)
output2 = Dense(64)(dense14)
#머지
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(32)(merge1)
merge3 = Dense(64)(merge2)
merge4 = Dense(64, activation='ELU')(merge3)
merge5 = Dense(32,activation='ELU')(merge4)
last_output = Dense(1)(merge5)

model = Model(inputs = [input1,input2], outputs = last_output)
# model.summary()

#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath = './_save/MCP/시험/'
filename = '{epoch:04d}-{val_loss:.2f}.hdf5'

import datetime
date = datetime.datetime.now() 
date = date.strftime("%m%d_%H%M")

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 30, restore_best_weights=True)
mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'min',verbose=1, save_best_only=True, filepath ="".join([filepath,'시험_',date,'_',filename]))
model.fit([x1_train, x2_train],y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es,mcp])

#평가 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss:',loss)

y_predict =np.round(model.predict([x1_test, x2_test]),2)

print(y_predict[-1:])



