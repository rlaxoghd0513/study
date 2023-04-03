import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

path_save = 'd:/study_data/_save/men_women_flow/'

x_train = np.load(path_save + 'keras58_10_x_train.npy')
x_test = np.load(path_save + 'keras58_10_x_test.npy')
y_train = np.load(path_save + 'keras58_10_y_train.npy')
y_test = np.load(path_save + 'keras58_10_y_test.npy')


print(x_train.shape,x_test.shape)
print(y_train.shape, y_test.shape)

# 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout,Flatten
from sklearn.metrics import accuracy_score

model = Sequential()
model.add(Conv2D(14, (3,3), padding='same', input_shape=(100,100,3)))
model.add(Conv2D(7, (3,3), padding='same'))
model.add(Conv2D(11, (4,4), padding='same'))
model.add(Flatten())
model.add(Dense(150, activation = 'relu'))
model.add(Dense(77))
model.add(Dense(1, activation = 'sigmoid'))


#3 컴파일 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=30, verbose=1, restore_best_weights=True)
model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 32)

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print('results:', loss)

y_predict = model.predict(x_test)


acc=accuracy_score(y_test, np.round(y_predict))
print('acc:',acc)