import numpy as np

path_save_train = 'd:/study_data/_save/project/배경x/'
path_save_test = 'd:/study_data/_save/project/test/'
path_save_predict = 'd:/study_data/_save/project/predict/'

x_train = np.load(path_save_train + 'project_x_train.npy')
y_train = np.load(path_save_train + 'project_y_train.npy')
x_test = np.load(path_save_test + 'project_x_test.npy')
y_test = np.load(path_save_test + 'project_y_test.npy')
x_predict = np.load(path_save_predict + 'project_x_predict.npy')
y_predict = np.load(path_save_predict + 'project_y_predict.npy')

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.metrics import accuracy_score

model = Sequential()
model.add(Conv2D(100, (2,2), input_shape = (100,100,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(100, 2, activation = 'relu'))
model.add(Conv2D(100, 2, activation = 'relu'))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation = 'softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_acc', mode = 'max', patience=40, restore_best_weights = True)
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_split= 0.1, callbacks = [es])

results = model.evaluate(x_test, y_test)
print('acc:', results[1])

y_predict_model = model.predict(x_predict)

y_predict = np.argmax(y_predict, axis=1)
y_predict_model = np.argmax(y_predict_model, axis=1)

print(y_predict,y_predict_model)

predict_acc = accuracy_score(y_predict, y_predict_model)
print('모델분류 정확도`:',predict_acc)
