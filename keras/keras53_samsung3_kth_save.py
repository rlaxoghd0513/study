import numpy as np
import pandas as pd

path = './_data/시험/'
path_save = './_save/시험/'

datasets1 = pd.read_csv(path + '삼성전자 주가2.csv', index_col = 0, encoding = 'utf-8', thousands = ',')
datasets2 = pd.read_csv(path + '현대자동차.csv', index_col =0, encoding = 'utf-8', thousands = ',')

print(datasets1.head)

datasets1 = datasets1.drop(['전일비'], axis=1)
datasets2 = datasets2.drop(['전일비'], axis=1)
datasets1 = datasets1.drop(['Unnamed: 6'], axis=1)
datasets2 = datasets2.drop(['Unnamed: 6'],axis=1)

print(type(datasets1)) #<class 'pandas.core.frame.DataFrame'>
print(type(datasets2)) #<class 'pandas.core.frame.DataFrame'>
print(datasets1.info())
print(datasets2.info())

datasets1 = datasets1.iloc[:1206]
datasets2 = datasets2.iloc[:1206]
###########################################################
datasets1 = datasets1.sort_values('일자', ascending=True)
datasets2 = datasets2.sort_values('일자', ascending=True)

x1 = datasets1.drop(['시가'], axis=1)
x2 = datasets2.drop(['시가'], axis=1)
y = datasets2['시가']

print(x1.isnull().sum())
print(x2.isnull().sum())



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
# print(x1)
print(x1.shape) #(1196, 10, 13)

x2 = split_x(x2, timesteps)
# print(x2)
print(x2.shape) #(1196, 10, 13)

y = y[timesteps+1:]
print(y.shape) #(1196,)
#########################################################


from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1,x2,y, random_state = 1520, train_size = 0.85)
print(x1_train.shape, x1_test.shape) #(1016, 10, 13) (180, 10, 13)
print(x2_train.shape, x2_test.shape) #(1016, 10, 13) (180, 10, 13)

x1_train = x1_train.reshape(1015,130)
x1_test = x1_test.reshape(180,130)
x2_train = x2_train.reshape(1015,130)
x2_test = x2_test.reshape(180,130)

from sklearn.preprocessing import MinMaxScaler, RobustScaler
scaler = RobustScaler()
x2_train = scaler.fit_transform(x2_train)
x2_test = scaler.transform(x2_test)

x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)

print(np.min(x1_train), np.max(x1_train))
print(np.min(x1_test), np.max(x1_test)) 
print(np.min(x2_train), np.max(x2_train))
print(np.min(x2_test), np.max(x2_test))

x1_train = x1_train.reshape(1015,10,13)
x1_test = x1_test.reshape(180,10,13)
x2_train = x2_train.reshape(1015,10,13)
x2_test = x2_test.reshape(180,10,13)



#모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input, Bidirectional, Flatten, Dropout, Dense
#모델1
input1 = Input(shape = (10,13))
lstm1  = Bidirectional(LSTM(64, return_sequences=True))(input1)
lstm2 = Bidirectional(LSTM(128, return_sequences=True))(lstm1)
lstm3 = LSTM(256)(lstm2)
dense1 = Dense(128)(lstm3)
dense2 = Dense(64)(dense1)
dense3 = Dense(128, activation = 'ELU')(dense2)
dense4 = Dense(64, activation = 'ELU')(dense3)
output1 = Dense(32)(dense4)
#모델2
input2 = Input(shape = (10,13))
lstm11 = LSTM(64, return_sequences=True)(input2)
lstm12 = Bidirectional(LSTM(128, return_sequences=True))(lstm11)
lstm13 = Bidirectional(LSTM(256))(lstm12)
dense11 = Dense(128)(lstm13)
dense12 = Dense(64)(dense11)
dense13 = Dense(128)(dense12)
dense14 = Dense(64, activation = 'ELU')(dense13)
output2 = Dense(32, activation='ELU')(dense14)
#머지
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(64)(merge1)
merge3 = Dense(128)(merge2)
merge4 = Dense(64)(merge3)
merge5 = Dense(32, activation='ELU')(merge4)
merge6 = Dense(16, activation='ELU')(merge5)
last_output = Dense(1)(merge6)

model = Model(inputs = [input1,input2], outputs = last_output)
model.summary()

#컴파일 훈련
model.compile(loss='mse', optimizer='adamax')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath = './_save/MCP/시험/'
filename = '{epoch:04d}-{val_loss:.2f}.hdf5'

import datetime
date = datetime.datetime.now() 
date = date.strftime("%m%d_%H%M")

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 70, restore_best_weights=True)
mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'min',verbose=1, save_best_only=True, filepath ="".join([filepath,'시험_',date,'_',filename]))
model.fit([x2_train, x1_train],y_train, epochs=10000, batch_size=8, validation_split=0.2, callbacks=[es,mcp])

#평가 예측
loss = model.evaluate([x2_test, x1_test], y_test)
print('loss:',loss)

from sklearn.metrics import r2_score
y_predict =np.round(model.predict([x2_test, x1_test]),2)
y_test = np.round(y_test,2)

r2 = r2_score(y_test, y_predict)
print('r2스코어:', r2)


print(y_predict[-1:])