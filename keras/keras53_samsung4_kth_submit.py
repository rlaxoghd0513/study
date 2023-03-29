import numpy as np
import pandas as pd

path = './_data/시험/'
path_save = './_save/시험/'

datasets1 = pd.read_csv(path + '삼성전자 주가3.csv', index_col = 0, encoding = 'utf-8', thousands = ',')
datasets2 = pd.read_csv(path + '현대자동차2.csv', index_col =0, encoding = 'utf-8', thousands = ',')


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


datasets1 = datasets1.iloc[:1207]
datasets2 = datasets2.iloc[:1207]

datasets1 = datasets1.sort_values('일자', ascending=True)
datasets2 = datasets2.sort_values('일자', ascending=True)


x1 = datasets1.drop(['시가'], axis=1)
x2 = datasets2.drop(['시가'], axis=1)
y = datasets2['시가']


timesteps = 10
def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps-1):  
        subset = dataset[i : (i+timesteps)] 
        aaa.append(subset)      
    return np.array(aaa)

x1 = split_x(x1, timesteps)

x2 = split_x(x2, timesteps)

y = y[timesteps+1:]



from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1,x2,y, random_state = 158796, train_size = 0.8)
# print(x1_train.shape, x1_test.shape) #(1076, 10, 9) (120, 10, 9)
# print(x2_train.shape, x2_test.shape) #(1076, 10, 9) (120, 10, 9)

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


from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Input, Bidirectional, Flatten, Dropout, Dense

model = load_model('_save\samsung\keras53_samsung4_kth.hdf5')

loss = model.evaluate([x1_test, x2_test], y_test)
print('loss:',loss)

y_predict =np.round(model.predict([x1_test, x2_test]),2)

print(y_predict[-1:])

