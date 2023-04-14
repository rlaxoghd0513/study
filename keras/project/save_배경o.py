Train_path = 'd:/study_data/_data/project/project/Training/'
Test_path = 'd:/study_data/_data/project/project/Validation/'
save_path_train = 'd:/study_data/_save/project/배경o/'
save_path_test = 'd:/study_data/_save/project/test/'

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
                                  shuffle = True)   #class_weights 각 라벨별로 이미지 개수를 균등하게 불러준다

Test_data = datagen.flow_from_directory(Test_path,
                                              target_size=(150,150),
                                              batch_size = 5000,
                                              class_mode = 'categorical',
                                              color_mode = 'rgb',
                                              shuffle=True)

x_train = Train_data[0][0]
y_train = Train_data[0][1]

x_test = Test_data[0][0]
y_test = Test_data[0][1]

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
print(np.min(x_test), np.max(x_test))

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)

np.save(save_path_train + 'x_train_o.npy', arr = x_train)
np.save(save_path_test + 'x_test.npy', arr = x_test)
np.save(save_path_train + 'y_train_o.npy', arr = y_train)
np.save(save_path_test + 'y_test.npy', arr = y_test)