train_path = 'd:/study_data/_data/project/project/Training/'
predict_path = 'd:/study_data/_data/project/project/Validation/'
save_path = 'd:/study_data/_save/project/'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(         
    horizontal_flip=True,   
    vertical_flip=True,    
    width_shift_range=0.1,  
    height_shift_range=0.1, 
    rotation_range=5,       
    zoom_range=1.2,        
    shear_range=0.7,        
    fill_mode='nearest')

Train_data = datagen.flow_from_directory(train_path,
                                  target_size = (150,150),
                                  batch_size = 130000, 
                                  class_mode = 'categorical',
                                  color_mode = 'rgb',
                                  shuffle=True)

Predict_data = datagen.flow_from_directory(predict_path,
                                              target_size=(150,150),
                                              batch_size = 5000,
                                              class_mode = 'categorical',
                                              color_mode = 'rgb',
                                              shuffle=True)

x = Train_data[0][0]
y = Train_data[0][1]

import numpy as np


x_predict = Predict_data[0][0]
y_predict = Predict_data[0][1]
print(x.shape, y.shape) # (100, 100, 100, 3) (100, 7) 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 333, train_size = 0.7)

x_train_1, x_train_augmented, y_train_1, y_train_augmented = train_test_split(x_train, y_train, random_state= 333, train_size = 0.8)

augment_size = x_train_augmented.shape[0]

x_train_augmented = train_datagen.flow(
    x_train_augmented, y_train_augmented,
    batch_size = augment_size,
    shuffle= False
).next()[0]

import numpy as np
x_train = np.concatenate((x_train_1, x_train_augmented))
y_train = np.concatenate((y_train_1, y_train_augmented))

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))
print(np.min(x_predict), np.max(x_predict))

print(x_train.shape, y_train.shape) #(4900, 100, 100, 3) (4900, 7)
print(x_test.shape, y_test.shape) #(2100, 100, 100, 3) (2100, 7)
print(x_predict.shape,y_predict.shape) #(70, 100, 100, 3) (70, 7)


np.save(save_path + 'project_x_train.npy', arr = x_train)
np.save(save_path + 'project_x_test.npy', arr = x_test)
np.save(save_path + 'project_x_predict.npy', arr = x_predict)
np.save(save_path + 'project_y_train.npy', arr = y_train)
np.save(save_path + 'project_y_test.npy', arr = y_test)
np.save(save_path + 'project_y_predict.npy', arr = y_predict)







