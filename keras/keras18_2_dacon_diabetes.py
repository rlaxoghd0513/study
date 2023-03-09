import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import pandas as pd

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

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1489, shuffle=True, train_size=0.9)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
#(521, 8) (131, 8)
#(521,) (131,)

#모델구성
model=Sequential()
model.add(Dense(15,input_dim=8))
model.add(Dense(12))
model.add(Dense(14))
model.add(Dense(18))
model.add(Dense(18))
model.add(Dense(18))
model.add(Dense(11,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es=EarlyStopping(monitor='val_loss', patience=20, mode='min',verbose=1, restore_best_weights=True)
model.fit(x_train,y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es])

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

submission.to_csv(path_save+'submit_0309_1434.csv')