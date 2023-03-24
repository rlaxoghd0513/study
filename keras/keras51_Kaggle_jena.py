import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, GRU, SimpleRNN, Conv1D
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1 데이터
path = './_data/kaggle_jena/'
pathsave ='./_save/kaggle_jena/'

filepath = './_save/MCP/kaggle_jena/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

import datetime
date = datetime.datetime.now() 
date = date.strftime("%m%d_%H%M%S") 
 

x_trains = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)  #날짜는 연산을 못해서 걍 뺀다
# print(x_trains)  #[420551 rows x 14 columns]   7대2대1  train test predict
# print(x_trains.columns)  
         #Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
    #     'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
    #     'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
    #     'wd (deg)'],
    #     dtype='object')
# print(x_trains.info()) #결측치 없다
# print(x_trains.describe())   #x_trains.head()는 다섯개만 보여준다

# print(x_trains['T (degC)'])  #데이터 형태는 판다스
# print(x_trains['T (degC)'].values)  #  .values 판다스를 넘파이로 바꾼다
# print(x_trains['T (degC)'].to_numpy())  #이것도 판다스를 넘파이로 바꾼다

# import matplotlib.pyplot as plt  #플롯 그림그리기는 넘파이 형태여야한다
# plt.plot(x_trains['T (degC)'].values)   
# plt.show()

x = x_trains.drop(['T (degC)'], axis =1)
y = x_trains['T (degC)']
print(x)
print(y)


x_train, x_test, y_train, y_test  = train_test_split(x,y, shuffle= False, train_size = 0.7)
print(x_train.shape, x_test.shape)  #(294385, 13) (126166, 13)
print(y_train.shape, y_test.shape)  # (294385,) (126166,)

x_test, x_predict, y_test, y_predict = train_test_split(x_test, y_test, shuffle = False, train_size = 0.67)
print(x_test.shape, x_predict.shape)  #(84531, 13) (41635, 13)
print(y_test.shape, y_predict.shape)  #(84531,) (41635,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_predict = scaler.transform(x_predict)

###############################################################################################################
timesteps = 10
def split_x(x_train, timesteps):
    aaa = []
    for i in range(len(x_train) - timesteps): 
        subset = x_train[i : (i+timesteps)] 
        aaa.append(subset)        
    return np.array(aaa)

 
x_train= split_x(x_train, timesteps)
x_test = split_x(x_test, timesteps)
x_predict = split_x(x_predict, timesteps)
print(x_train.shape)   #(294375, 10, 13)
print(x_test.shape)    #(84521, 10, 13)
print(x_predict.shape) #(41625, 10, 13)

y_train = y_train[timesteps:]
y_test = y_test[timesteps:]
y_predict = y_predict[timesteps:]
print(y_train.shape)
print(y_test.shape)
print(y_predict.shape)


#모델구성
model = Sequential()
model.add(Bidirectional(LSTM(32, return_sequences = True), input_shape = (10,13)))
model.add(Bidirectional(LSTM(64, return_sequences = True)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(LSTM(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16,activation = 'ELU'))
model.add(Dense(1, activation ='ELU'))

#컴파일
model.compile(loss = 'mse', optimizer= 'adam')
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience =50, restore_best_weights=True)
mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'min', save_best_only=True, filepath ="".join([filepath,'jena_',date,'_',filename]))
model.fit(x_train, y_train, epochs = 10000, batch_size = 32, validation_split = 0.2, callbacks = [es,mcp])

results = model.evaluate(x_test, y_test)
print('results:', results)

y_predict_m = model.predict(x_predict)

def RMSE(y_predict, y_predict_m):     
    return np.sqrt(mean_squared_error(y_predict, y_predict_m)) 
rmse = RMSE(y_predict, y_predict_m)   
print("RMSE :", rmse)

r2 = r2_score(y_predict, y_predict_m)
print('r2스코어:', r2)












