import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

path_save = 'd:/study_data/_save/dogs_breed/'

db_x_train = np.load(path_save + 'keras56_db_x_train.npy')
db_x_test = np.load(path_save + 'keras56_db_x_test.npy')
db_y_train = np.load(path_save + 'keras56_db_y_train.npy')
db_y_test = np.load(path_save + 'keras56_db_y_test.npy')

#모델구성
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape = (200,200,3), activation = 'relu'))
model.add(Conv2D(64,3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(16 , activation = 'relu'))
model.add(Dense(5, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(db_x_train, db_y_train, epochs = 100, validation_data = (db_x_test, db_y_test))

loss = model.evaluate(db_x_test, db_y_test)
print('loss:', loss)

y_predict = model.predict(db_x_test)
from sklearn.metrics import accuracy_score

y_test = np.argmax(db_y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test, y_predict)
print('acc:', acc)
