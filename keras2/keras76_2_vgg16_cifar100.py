import numpy as np
from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights = 'imagenet',
              include_top = False,#True로 하면 (224,224,3) 이 인풋이여야 한다
              input_shape = (32,32,3)
              )

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

vgg16.trainable = True

model = Sequential()
model.add(vgg16)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(32))
model.add(Dense(100, activation = 'softmax'))

model.summary()

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=5, verbose=1, restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics = ['accuracy'])

import time
start_time = time.time()

model.fit(x_train, y_train, epochs = 100, batch_size = 32, callbacks = [es], validation_split=0.2)
end_time = time.time()

#4 평가 예측
results = model.evaluate(x_test, y_test)
print('results:', results)
print('걸린시간:', round(end_time - start_time,2))

y_predict = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)
y_predict_acc = np.argmax(y_predict, axis=1)

from sklearn.metrics import accuracy_score

acc=accuracy_score(y_test_acc, y_predict_acc)
print('acc:',acc)
