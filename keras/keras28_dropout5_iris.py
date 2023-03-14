import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping

#1데이터
datasets = load_iris()
print(datasets.DESCR) 


x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)

print(x)
print(y)
print('y의 라벨값:', np.unique(y))   #y의 라벨값 [0,1,2]


from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) #(150, 3)


x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, 
                                                    random_state=333,
                                                    train_size=0.8,
                                                    stratify=y 
                                                    )
print(y_train)
print(np.unique(y_train, return_counts=True)) 

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))


input1 = Input(shape=(4, ))
dense1 = Dense(50)(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(40)(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(40)(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(10)(drop3)
output1 = Dense(3)(dense4)
model = Model(inputs = input1, outputs = output1)

#3 컴파일 훈련
es = EarlyStopping(monitor='acc', patience = 20, mode='max'
              ,verbose=1
              ,restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])  #다중분류에서 loss는 
model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split = 0.2, verbose=1)

#4. 평가 예측
results = model.evaluate(x_test, y_test) 
print(results)    
print('loss:', results[0])
print('acc:',results[1])

y_predict = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis=1)  
y_pred = np.argmax(y_predict, axis = 1) 
acc = accuracy_score(y_test_acc, y_pred) 

print('acc:', acc)

# [1.075387716293335, 0.3333333432674408]
# [2.6863491535186768, 0.3333333432674408]