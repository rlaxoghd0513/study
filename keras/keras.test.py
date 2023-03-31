import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
path = 'd:/study_data/_data/rsp/'
save_path = 'd:/study_data/_save/rsp/'

all_data = ImageDataGenerator(
    rescale=1./255
)

start = time.time()

rps_data = all_data.flow_from_directory(
    path,
    target_size=(100, 100),
    batch_size=100,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

x_rps = rps_data[0][0]
y_rps = rps_data[0][1]

end = time.time()

print(x_rps.shape, y_rps.shape) # (100, 100, 100, 3) (100, 3)

print(end - start)

x_train, x_test, y_train, y_test = train_test_split(
    x_rps,
    y_rps,
    test_size=0.025,
    random_state=2222,
    shuffle=True
)

np.save(save_path + 'keras56_rps_x_train.npy', arr=x_train)
np.save(save_path + 'keras56_rps_x_test.npy', arr=x_test)
np.save(save_path + 'keras56_rps_y_train.npy', arr=y_train)
np.save(save_path + 'keras56_rps_y_test.npy', arr=y_test)

print(x_train.shape, x_test.shape) # (97, 100, 100, 3) (3, 100, 100, 3)
print(y_train.shape, y_test.shape) # (97,) (3,)

#2. MODEL
model = Sequential()
model.add(Conv2D(128, (2, 2), input_shape=(100, 100, 3), activation='relu'))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))


#3. COMPILE
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
Es = EarlyStopping(
    monitor='val_acc',
    patience=50,
    mode='auto',
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[Es])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:', val_acc[-1])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.subplot(1, 2, 1)
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(acc, label='acc')
plt.plot(val_acc, label='val_acc')
plt.grid()
plt.legend()
plt.show()

# loss: 3.2749852834967896e-05
# val_loss: 0.06971368938684464
# acc: 1.0
# val_acc: 1.0