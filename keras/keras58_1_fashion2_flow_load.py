import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

path_save = 'd:/study_data/_save/fashion_mnist/'

x_train = np.load(path_save + 'keras_58_1_fashion_flow_x_train.npy')
y_train = np.load(path_save + 'keras_58_1_fashion_flow_y_train.npy')
x_test = np.load(path_save + 'keras_58_1_fashion_flow_x_test.npy')
y_test = np.load(path_save + 'keras_58_1_fashion_flow_y_test.npy')

from tensorflow.keras.utils import to_categorical
 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout,Flatten
from sklearn.metrics import accuracy_score

model = Sequential()
model.add(Conv2D(14, (3,3), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(7, (3,3), padding='same'))
model.add(Conv2D(11, (4,4), padding='same'))
model.add(Conv2D(10, (3,3), padding = 'same'))
model.add(Flatten())
model.add(Dense(280, activation = 'relu'))
model.add(Dense(200))
model.add(Dense(150, activation = 'relu'))
model.add(Dense(77))
model.add(Dense(10, activation = 'softmax'))


#3 컴파일 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=30, verbose=1, restore_best_weights=True)
model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 10000, batch_size = 128, callbacks = [es], validation_split =0.2)

#4 평가 예측
results = model.evaluate(x_test, y_test)
print('results:', results)

y_predict = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)
y_predict_acc = np.argmax(y_predict, axis=1)

acc=accuracy_score(y_test_acc, y_predict_acc)
print('acc:',acc)