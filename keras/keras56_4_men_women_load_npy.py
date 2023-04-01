import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

path_save = 'd:/study_data/_save/men_women/'

mw_x_train = np.load(path_save + 'keras56_mw_x_train.npy')
mw_x_test = np.load(path_save + 'keras56_mw_x_test.npy')
mw_y_train = np.load(path_save + 'keras56_mw_y_train.npy')
mw_y_test = np.load(path_save + 'keras56_mw_y_test.npy')

#모델구성
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(200, 200, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(mw_x_train, mw_y_train, epochs=100,validation_data = (mw_x_test, mw_y_test))

loss = model.evaluate(mw_x_test, mw_y_test)
print('loss:', loss)

y_predict = model.predict(mw_x_test)
from sklearn.metrics import accuracy_score

acc = accuracy_score(mw_y_test,np.round(y_predict) )
print('acc:', acc)