import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from sklearn.model_selection import train_test_split

path = 'd:/study_data/_data/horse_or_human/'
path_save = 'd:/study_data/_save/horse_human/'

datagen = ImageDataGenerator(rescale=1./255)
datagen2 = ImageDataGenerator(rescale=1./255.,
                              horizontal_flip=True,
                              height_shift_range = 0.1,
                              fill_mode = 'nearest')

hoh = datagen.flow_from_directory(path,
                                  target_size = (150,150),
                                  batch_size = 100, 
                                  class_mode = 'binary',
                                  color_mode = 'rgb')

hoh_x = hoh[0][0]
hoh_y = hoh[0][1]

hoh_x_train, hoh_x_test, hoh_y_train, hoh_y_test = train_test_split(hoh_x, hoh_y, train_size = 0.7, random_state = 333, shuffle=True)
print(hoh_x_train.shape, hoh_x_test.shape)#(70, 150, 150, 3) (30, 150, 150, 3)


augment_size = 30
np.random.seed(42)
randint = np.random.randint(70, size = 30)
x_augmented = hoh_x_train[randint].copy()
y_augmented = hoh_y_train[randint].copy()

x_augmented = datagen2.flow(
    x_augmented, y_augmented, 
    batch_size = augment_size,
    shuffle = False
).next()[0]

print(x_augmented.shape)#(30, 150, 150, 3)

x_train = np.concatenate((hoh_x_train, x_augmented))
y_train = np.concatenate((hoh_y_train, y_augmented))
x_test = hoh_x_test
y_test = hoh_y_test
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

np.save(path_save + 'keras58_6_hoh_x_train.npy', arr=x_train)
np.save(path_save + 'keras58_6_hoh_x_test.npy', arr=x_test)
np.save(path_save + 'keras58_6_hoh_y_train.npy', arr=y_train)
np.save(path_save + 'keras58_6_hoh_y_test.npy', arr=y_test)
