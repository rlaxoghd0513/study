import numpy as np

path_save = 'd:/study_data/_save/project/'

x_train = np.load(path_save + 'project_x_train.npy')
x_test = np.load(path_save + 'project_x_test.npy')
y_train = np.load(path_save + 'project_y_train.npy')
y_test = np.load(path_save + 'project_y_test.npy')
x_predict = np.load(path_save + 'project_x_predict.npy')
y_predict = np.load(path_save + 'project_y_predict.npy')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D
from sklearn.metrics import accuracy_score

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (100,100,3)))
model.add(Conv2D(16,2))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(7, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 1)

loss = model.evaluate(x_test, y_test)
print(loss)