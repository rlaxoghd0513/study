from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import time
path = 'd:/study_data/_data/cat_dog/PetImages/'
save_path = 'd:/study_data/_save/cat_dog/'

datagen = ImageDataGenerator(rescale=1./255)
start = time.time()
cat_dog = datagen.flow_from_directory('d:/study_data/_data/cat_dog/PetImages/', target_size=(250, 250), batch_size=1000, class_mode='binary', color_mode='rgb', shuffle=True)
#(25, 250, 250, 3)
cat_dog_x = cat_dog[0][0]
cat_dog_y = cat_dog[0][1]

end = time.time()
print(end - start)
print(cat_dog[0][0].shape, cat_dog[:24][0].shape) #(1000, 250, 250, 3) (998, 250, 250, 3)
print(cat_dog[0][1].shape, cat_dog[24][1].shape)
cat_dog_x_train, cat_dog_x_test, cat_dog_y_train, cat_dog_y_test = train_test_split(cat_dog_x, cat_dog_y, train_size=0.7, shuffle=True, random_state=123)
print(cat_dog[0][1])

print(cat_dog_x_train.shape)
print(cat_dog_x_test.shape)
print(cat_dog_y_train.shape)
print(cat_dog_y_test.shape)

np.save(save_path + 'keras56_cat_dog_x_train.npy', arr=cat_dog_x_train)
np.save(save_path + 'keras56_cat_dog_x_test.npy', arr=cat_dog_x_test)
np.save(save_path + 'keras56_cat_dog_y_train.npy', arr=cat_dog_y_train)
np.save(save_path + 'keras56_cat_dog_y_test.npy', arr=cat_dog_y_test)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(250, 250, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(cat_dog_x_train, cat_dog_y_train, epochs=100, validation_data=(cat_dog_x_test, cat_dog_y_test))

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

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








