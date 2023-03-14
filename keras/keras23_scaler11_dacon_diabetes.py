import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
scaler= StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))


test_csv = scaler.transform(test_csv)  #train_csv도 스케일링 했으니까 제출할 test_csv 도 마찬가지로 스케일링 해줘야한다


#모델구성
model=Sequential()
model.add(Dense(10,input_dim=8))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es=EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
model.fit(x_train,y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es])

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

submission.to_csv(path_save+'submit_0313_1153.csv')

#minmax loss = 0.52  acc = 0.72
#standard loss = 0.48 acc = 0.83
#robust loss= 0.52 acc = 0.6
#maxabs loss= 0.51 acc=0.72
