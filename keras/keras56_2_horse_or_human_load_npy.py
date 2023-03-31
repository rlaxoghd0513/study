import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

path_save = 'd:/study_data/_save/horse_or_human/'

hoh_x_train = np.load(path_save + 'keras56_hoh_x_train.npy')
hoh_x_test = np.load(path_save + 'keras56_hoh_x_test.npy')
hoh_y_train = np.load(path_save + 'keras56_hoh_y_train.npy')
hoh_y_test = np.load(path_save + 'keras56_hoh_y_test.npy')

print(hoh_x_train.shape, hoh_x_test.shape)  #(70, 150, 150, 3) (30, 150, 150, 3)
print(hoh_y_train.shape, hoh_y_test.shape)  #(70,) (30,)
print(hoh_y_train)

#모델
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(hoh_x_train, hoh_y_train, epochs = 100, validation_data = (hoh_x_test,hoh_y_test))

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

loss = model.evaluate(hoh_x_test, hoh_y_test)
print('loss:', loss)
y_predict = model.predict(hoh_x_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(hoh_y_test, np.round(y_predict))