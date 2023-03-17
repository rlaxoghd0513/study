import numpy as np
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


train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
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

model = load_model('./_save/MCP/call/call_0317_163409_0453-0.3214.hdf5')

results = model.evaluate(x_test, y_test)
print('results:', results)
y_test_acc = np.argmax(y_test, axis=1)
y_predict = model.predict(x_test)
y_predict_acc = np.argmax(y_predict, axis=1)

acc=accuracy_score(y_test_acc, y_predict_acc)
print('acc:',acc)
f1 = f1_score(y_test_acc, y_predict_acc, average='macro')
print('f1:', f1)


y_submit = model.predict(test_csv) #submit 제출
# print(y_submit)
y_submit = np.argmax(y_submit, axis=1)

submission = pd.read_csv(path+'sample_submission.csv',index_col=0)

                    
# print(submission)
submission[train_csv.columns[-1]] = y_submit
# print(submission)


submission.to_csv(pathsave+'call_0317_1722.csv')
