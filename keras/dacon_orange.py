import numpy as np
from tensorflow.python.keras.models import Model, load_model, Sequential
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler, MinMaxScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

path = './_data/dacon_orange/'
pathsave ='./_save/dacon_orange/'

filepath = './_save/MCP/dacon_orange/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

import datetime
date = datetime.datetime.now() 
date = date.strftime("%m%d_%H%M%S") 

train_csv = pd.read_csv(path+'train.csv', index_col=0)
print(train_csv.shape) #(2207, 183)

test_csv = pd.read_csv(path+'test.csv', index_col=0)
print(test_csv.shape) #(2208, 182)

print(train_csv.info()) #non-null

x = train_csv.drop(['착과량(int)'], axis=1)
print(x.shape)  #(2207, 182)
y = train_csv['착과량(int)']
print(y.shape)  #(2207,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, shuffle=True, random_state = 1234)
print(x_train.shape, x_test.shape)   #(1765, 182) (442, 182)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

model = Sequential()
model.add(Dense(64, input_dim =182))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(32, activation='ELU'))
model.add(Dense(1, activation='ELU'))

model.compile(loss = 'mse', optimizer='adam')
es = EarlyStopping(monitor = 'val_loss', mode ='min', restore_best_weights=True, patience = 50)
mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'min', save_best_only=True, filepath ="".join([filepath,'orange_',date,'_',filename]))
model.fit(x_train, y_train, epochs = 10000, batch_size = 32, validation_split =0.2, callbacks = [es,mcp])

loss= model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict = model.predict(x_test)
print(y_test.shape)    #(442,)
print(y_predict.shape)  #(442, 1)
y_predict = y_predict.reshape(-1)
print(y_predict.shape) #(442,)



def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score
nmae = NMAE(y_test, y_predict)
print('NMAE:', nmae)

y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + 'sample_submission.csv', index_col =0)

# print(submission)
submission['착과량(int)'] = y_submit
# print(submission)



submission.to_csv(pathsave+ 'orange_'+ date + '.csv')
