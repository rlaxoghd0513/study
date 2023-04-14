import numpy as np

path_save_train = 'd:/study_data/_save/project/배경x/'
path_save_test = 'd:/study_data/_save/project/test/'
path_save_predict = 'd:/study_data/_save/project/predict/'

x_train = np.load(path_save_train + 'x_train_x.npy')
y_train = np.load(path_save_train + 'y_train_x.npy')
x_test = np.load(path_save_test + 'x_test.npy')
y_test = np.load(path_save_test + 'y_test.npy')
x_predict = np.load(path_save_predict + 'x_predict.npy')
y_predict = np.load(path_save_predict + 'y_predict.npy')

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.regularizers import L1, L2
from sklearn.metrics import accuracy_score

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (150,150,3), padding = 'same',activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding = 'same',activation = 'relu', kernel_regularizer = L1(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_regularizer = L1(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_regularizer = L2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(7, activation = 'softmax'))
model.summary()


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=50, restore_best_weights = True)
model.fit(x_train, y_train, epochs = 1000, batch_size = 4, validation_split= 0.1, callbacks = [es])

results = model.evaluate(x_test, y_test)
print('acc:', results[1])

y_predict_model = model.predict(x_predict)

y_predict = np.argmax(y_predict, axis=1)
y_predict_model = np.argmax(y_predict_model, axis=1)

print(y_predict,y_predict_model)

predict_acc = accuracy_score(y_predict, y_predict_model)
print('모델분류 정확도:', predict_acc)