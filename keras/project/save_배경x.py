Train_path = 'd:/study_data/_data/project/project/Training_x/'
save_path_train = 'd:/study_data/_save/project/배경x/'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(         
    horizontal_flip=True,   
    vertical_flip=True,        
    zoom_range=1.2)

Train_data = datagen.flow_from_directory(Train_path,
                                  target_size = (150,150),
                                  batch_size = 7000, 
                                  class_mode = 'categorical',
                                  color_mode = 'rgb',
                                  shuffle=True)


x_train = Train_data[0][0]
y_train = Train_data[0][1]

augment_size = 5000
np.random.seed(42)
randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = train_datagen.flow(
    x_augmented, y_augmented, batch_size = augment_size, shuffle=False
).next()[0]


x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(np.min(x_train), np.max(x_train))

print(x_train.shape, y_train.shape) 

np.save(save_path_train + 'x_train_x.npy', arr = x_train)
np.save(save_path_train + 'y_train_x.npy', arr = y_train)











