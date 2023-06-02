#[실습] keras56_4 남자 여자에 noise를 넣는다
#predict : 기미 주근깨 제거
#5개 사진 출력 / 원본, 노이즈, 아웃풋

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D
from matplotlib import pyplot as plt
import random
import tensorflow as tf

path = 'd:/men_women/'
save_path = 'd:/_save/'

all_data = ImageDataGenerator(
    rescale=1./255
)

start = time.time()

people_data = all_data.flow_from_directory(
    path,
    target_size=(100, 100),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

x_data = people_data[0][0]
y_data = people_data[0][1]

end = time.time()

print(end - start)

x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    test_size=0.025,
    random_state=2222,
    shuffle=True
)

print(x_train.shape, x_test.shape) # (97, 100, 100, 3) (3, 100, 100, 3)
print(y_train.shape, y_test.shape) # (97,) (3,)


# Add noise
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

# Model configuration
model = Sequential()
# Encode
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(100, 100, 3)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

# Decode
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

model.summary()

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(x_train_noised, x_train, epochs=100, batch_size=32)

# Generate decoded images
decoded_imgs = model.predict(x_test_noised)

# Randomly select images for visualization
num_images = min(5, decoded_imgs.shape[0])
random_images = random.sample(range(decoded_imgs.shape[0]), num_images)

fig, axes = plt.subplots(3, num_images, figsize=(20, 7))

for i, ax in enumerate(axes[0]):
    ax.imshow(x_test[random_images[i]].reshape(100, 100, 3), cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate(axes[1]):
    ax.imshow(x_test_noised[random_images[i]].reshape(100, 100, 3), cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate(axes[2]):
    ax.imshow(decoded_imgs[random_images[i]].reshape(100, 100, 3), cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()