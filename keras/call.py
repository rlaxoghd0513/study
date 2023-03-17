import numpy as np   #1#2#@#@#@#@#@#@#@#@#@#@#@#
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler, MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score

path = './_data/call/'
pathsave = './_save/call/'

filepath = './_save/MCP/call/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)

print(train_csv.shape)   #(30200, 13)
print(test_csv.shape)    #(12943, 12)
print(train_csv.info())    #non_null

x = train_csv.drop([train_csv.columns[-1]], axis=1)
y = train_csv[train_csv.columns[-1]]

y = to_categorical(y)
print(y.shape)    #(30200, 2)

print('y라벨값:', np.unique(y))  # y라벨값: [0 1]

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 1234, shuffle=True, train_size = 0.8, stratify = y)

scaler= RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

test_csv = scaler.transform(test_csv)  #train_csv도 스케일링 했으니까 제출할 test_csv 도 마찬가지로 스케일링 해줘야한다


input1 = Input(shape = (12,))
dense1 = Dense(16)(input1)
drop1 = Dropout(0.25)(dense1)
dense2 = Dense(32)(drop1)
drop2 = Dropout(0.25)(dense2)
dense3 = Dense(16)(drop2)
dense4 = Dense(32)(dense3)
dense5 = Dense(16, activation = 'relu')(dense4)
dense6 = Dense(16, activation='relu')(dense5)
output1 = Dense(2,activation= 'sigmoid' )(dense6)
model = Model(inputs = input1, outputs = output1)
model.summary()

#컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import datetime
date = datetime.datetime.now() 
date = date.strftime("%m%d_%H%M%S") 
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=50, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'min', save_best_only=True, filepath="".join([filepath,'call_',date,'_',filename]))
model.fit(x_train,y_train, epochs=10000, batch_size=64, validation_split=0.2, callbacks = [es,mcp])


#평가 에측
results = model.evaluate(x_test, y_test)
print('results:', results)
y_predict = np.argmax(model.predict(x_test))

acc=accuracy_score(y_test, y_predict)
print('acc:',acc)
f1 = f1_score(y_test, y_predict, average='macro')
print('f1:', f1)


y_submit = model.predict(test_csv) #submit 제출
# print(y_submit)
y_submit = np.argmax(y_submit, axis=1)

submission = pd.read_csv(path+'sample_submission.csv',index_col=0)

                    
# print(submission)
submission['전화해지여부'] = y_submit
# print(submission)


submission.to_csv(pathsave+ 'call_'+ date + '.csv')









