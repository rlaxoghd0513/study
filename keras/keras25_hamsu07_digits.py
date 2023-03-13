#사이킷런 load_digits
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(1797, 64) (1797,)

print(x)
print(y)
print('y의 라벨값:', np.unique(y)) #[0 1 2 3 4 5 6 7 8 9]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) #(1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=333, train_size=0.8, stratify=y )
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))


print(y_train)
print(np.unique(y_train, return_counts=True))


# model = Sequential()
# model.add(Dense(80, activation = 'relu',input_dim=64))
# model.add(Dense(70, activation = 'relu'))
# model.add(Dense(60, activation = 'relu'))
# model.add(Dense(70, activation = 'relu'))
# model.add(Dense(10, activation = 'softmax'))

input1 = Input(shape=(64, ))
dense1 = Dense(80, activation ='relu')(input1)
dense2 = Dense(70, activation = 'relu')(dense1)
dense3 = Dense(60, activation = 'relu')(dense2)
dense4 = Dense(70, activation = 'relu')(dense3)
output1 = Dense(10, activation = 'softmax')(dense4)
model = Model(inputs = input1,  outputs = output1)


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc']) 
es= EarlyStopping(monitor = 'acc', mode= 'max', patience= 20, verbose= 1, restore_best_weights =True) 
model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split = 0.2, verbose=1, callbacks= [es])

results = model.evaluate(x_test, y_test)
print('results:', results)

y_predict = model.predict(x_test)
print(y_predict)
y_test_acc = np.argmax(y_test, axis=1)
y_predict_acc = np.argmax(y_predict, axis=1)

acc=accuracy_score(y_test_acc, y_predict_acc)
print('acc:',acc)

#sequential acc  0.9916666666666667
#model      acc  0.9805555555555555