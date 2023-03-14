import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1데이터
datasets = load_breast_cancer()
print(datasets.feature_names) 

x=datasets['data']
y=datasets.target

print(x.shape, y.shape)  #(569, 30) (569,)
# print(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 123, train_size=0.8, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))


input1 = Input(shape = (30,))
dense1 = Dense(10, activation='linear')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(9)(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(8)(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(7)(drop3)
output1 = Dense(1, activation = 'sigmoid')(dense4)
model = Model(inputs = input1, outputs = output1)

#3 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics= ['accuracy','mse'] 
              ) 
from tensorflow.python.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor= 'val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True )
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1, callbacks=[es])

#4 평가 예측
results = model.evaluate(x_test, y_test)  
print('results:', results)
y_predict= np.round(model.predict(x_test))

# 정확도 accurate
from sklearn.metrics import r2_score, accuracy_score
acc=accuracy_score(y_test, y_predict)
print('acc:', acc)  

# results: [0.09117162972688675, 0.9824561476707458, 0.016791678965091705]
# results: [0.09954003244638443, 0.9649122953414917, 0.026690484955906868]