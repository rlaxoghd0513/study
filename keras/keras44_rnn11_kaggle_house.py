import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import MaxPool2D, Dense, Input, Dropout, Flatten, Conv2D,  GRU, SimpleRNN, LSTM, Bidirectional
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

#1 데이터
path = './_data/kaggle_house/'
path_save = './_save/kaggle_house/'

train_csv = pd.read_csv(path+ 'train.csv', index_col=0)

# print(train_csv.shape)  #(1460, 80)

train_csv = train_csv.drop(['LotFrontage'], axis=1)

test_csv = pd.read_csv(path+ 'test.csv', index_col=0)
test_csv = test_csv.drop(['LotFrontage'], axis=1)
# print(test_csv.shape)   #(1459, 79)

string_cols = train_csv.select_dtypes(include=['object']).columns  # 문자열인 칼럼을 찾는다
numeric_cols = train_csv.select_dtypes(exclude=['object'])  #문자열인 칼럼 제외한 나머지 선택
le = LabelEncoder()
string_cols_le = train_csv[string_cols].apply(lambda x: le.fit_transform(x))

train_csv  = pd.concat([numeric_cols, string_cols_le], axis=1)

print(train_csv.shape)  #  (1460, 79)

string_cols = test_csv.select_dtypes(include=['object']).columns  # 문자열인 칼럼을 찾는다
numeric_cols = test_csv.select_dtypes(exclude=['object'])  #문자열인 칼럼 제외한 나머지 선택
le = LabelEncoder()
string_cols_le = test_csv[string_cols].apply(lambda x: le.fit_transform(x))

test_csv  = pd.concat([numeric_cols, string_cols_le], axis=1)
print(test_csv.shape)   #(1459, 78)


print(train_csv.isnull().sum())
print(test_csv.isnull().sum())

means = train_csv.mean()
train_csv = train_csv.fillna(means)
print(train_csv.isnull().sum())

x = train_csv.drop(['SalePrice'], axis =1)
y = train_csv['SalePrice']

means = test_csv.mean()
test_csv = test_csv.fillna(means)
#########################################################################################################


x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 8739, shuffle = True, train_size=0.9)
print(x_train.shape, x_test.shape)   #(1314, 78) (146, 78)
print(y_train.shape, y_test.shape)     #(1314,) (146,)
print(test_csv.shape)   #(1459,78)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

x_train = x_train.reshape(1314,13,6)
x_test = x_test.reshape(146,13,6)

test_csv = test_csv.reshape(1459,13,6)



#2 모델구성
# input1 = Input(shape = (13,6))
# conv1 = LSTM(64, return_sequences=True)(input1)
# conv2 = LSTM(64,return_sequences = True)(conv1)
# conv3 = LSTM(128)(conv2)
# dense1 = Dense(128)(conv3)
# drop1 = Dropout(0.4)(dense1)
# dense2 = Dense(64)(drop1)
# dense3= Dense(32)(dense2)
# drop2 = Dropout(0.25)(dense3)
# dense4 = Dense(16, activation='ELU')(drop2)
# output1 = Dense(1, activation='ELU')(dense4)
# model = Model(inputs = input1, outputs=output1)

model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape = (13,6)))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Dense(32, activation = 'ELU'))
model.add(Dense(1, activation='ELU'))





#컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es= EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
filepath = './_save/MCP/kaggle_house/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
import datetime
date= datetime.datetime.now()
date = date.strftime("%m%d_%H%M%S")
mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, filepath ="".join([filepath,'house_',date,'_',filename]))
model.fit(x_train, y_train, epochs = 10000, callbacks = [es,mcp], validation_split= 0.2, batch_size=32)

#평가 예측
results = model.evaluate(x_test,y_test)
print('results:', results)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):    
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
rmse = RMSE(y_test, y_predict)  
print("RMSE :", rmse)



y_submit = model.predict(test_csv) #submit 제출
print(y_submit)

submission = pd.read_csv(path+'sample_submission.csv',index_col=0)
                    

submission['SalePrice'] = y_submit


submission.to_csv(path_save+'kaggle_house_'+date+'.csv')

#results: 837226880.0
# RMSE : 28934.873384576436