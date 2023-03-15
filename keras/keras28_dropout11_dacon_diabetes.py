import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input,Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import RobustScaler,StandardScaler, MaxAbsScaler, MinMaxScaler
#데이터
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
print(train_csv.shape) #(652, 9)

test_csv = pd.read_csv(path+'test.csv', index_col=0)
print(test_csv.shape) #(116, 8)

print(train_csv.info()) #non-null

x = train_csv.drop(['Outcome'], axis=1)
print(x) #[652 rows x 8 columns]

y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1234, shuffle=True, train_size=0.9)
# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
#(521, 8) (131, 8)
#(521,) (131,)
scaler= RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))


test_csv = scaler.transform(test_csv)  #train_csv도 스케일링 했으니까 제출할 test_csv 도 마찬가지로 스케일링 해줘야한다


input1 = Input(shape = (8,))
dense1 = Dense(10)(input1)
drop1=Dropout(0.2)(dense1)
dense2 = Dense(8)(drop1)
dense3 = Dense(7)(dense2)
dense4 = Dense(6)(dense3)
drop4 = Dropout(0.1)(dense4)
dense5 = Dense(5, activation = 'relu')(drop4)
dense6 = Dense(2, activation='relu')(dense5)
output1 = Dense(1,activation= 'sigmoid' )(dense6)
model = Model(inputs = input1, outputs = output1)

#컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
import datetime
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M") 
 

filepath = './_save/MCP/dacon_diabetes/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

es=EarlyStopping(monitor = 'val_loss', patience=60, mode='min', verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor = 'val_loss', mode= 'auto', save_best_only=True, verbose=1, filepath ="".join([filepath,'dia_',date,'_',filename]))

model.fit(x_train,y_train, epochs=10000, batch_size=4, validation_split=0.2, callbacks=[es,mcp])


#평가 에측
results = model.evaluate(x_test, y_test)
print('results:', results)
y_predict = np.round(model.predict(x_test))

acc=accuracy_score(y_test, y_predict)
print('acc:',acc)

#서밋
y_submit = np.round(model.predict(test_csv)) #submit 제출
# print(y_submit)

submission = pd.read_csv(path+'sample_submission.csv',index_col=0)
                    
# print(submission)
submission['Outcome'] = y_submit
# print(submission)

submission.to_csv(path_save+'submit_0315_1428.csv')

# results: [0.4140123128890991, 0.8030303120613098]
# results: [0.41638749837875366, 0.8181818127632141]

#robust  results: [0.40475308895111084, 0.8181818127632141]
# standard  results: [0.43816661834716797, 0.8333333134651184]
# minmax  results: [0.6384103894233704, 0.6666666865348816]
#maxabs results: [0.4153611361980438, 0.8181818127632141]