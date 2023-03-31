import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

path_save = 'd:/study_data/_save/rsp/'

rsp_x_train = np.load(path_save + 'keras56_rsp_x_train.npy')
rsp_x_test = np.load(path_save + 'keras56_rsp_x_test.npy')
rsp_y_train = np.load(path_save + 'keras56_rsp_y_train.npy')
rsp_y_test = np.load(path_save + 'keras56_rsp_y_test.npy')

print(rsp_x_train.shape, rsp_x_test.shape) #(72, 100, 100, 3) (28, 100, 100, 3)
print(rsp_y_train.shape, rsp_y_test.shape) #(72, 3) (28, 3)
print(np.unique(rsp_y_train))#[0. 1.]

#모델구성
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(100, 100, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(rsp_x_train, rsp_y_train, epochs=100,validation_data = (rsp_x_test, rsp_y_test))

loss = model.evaluate(rsp_x_test, rsp_y_test)
print('loss:', loss)

y_predict = model.predict(rsp_x_test)
from sklearn.metrics import accuracy_score
rsp_y_test_acc = np.argmax(rsp_y_test, axis=1)
y_predict_acc = np.argmax(y_predict, axis=1)
acc = accuracy_score(rsp_y_test_acc, y_predict_acc)
print('acc:', acc)