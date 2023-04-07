import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from sklearn.model_selection import train_test_split

path = 'd:/study_data/_data/rsp/'
path_save = 'd:/study_data/_save/rsp_flow/'

datagen = ImageDataGenerator(rescale = 1./255)
datagen2 = ImageDataGenerator(horizontal_flip=True,
                              height_shift_range=0.1,
                              fill_mode= 'nearest')
rsp = datagen.flow_from_directory(path, target_size = (100,100), batch_size = 100, class_mode = 'categorical', color_mode = 'rgb', shuffle = True)

x = rsp[0][0]
y = rsp[0][1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.725, random_state = 333, shuffle=True)

print(x_train.shape, x_test.shape)#(72, 100, 100, 3) (28, 100, 100, 3)    #하위폴더에 있는 폴더 갯수만큼 카테고리컬 된다
print(y_train.shape, y_test.shape)#(72, 3) (28, 3)
    
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

augment_size = 28
np.random.seed(42)
randint = np.random.randint(72,size=28)
x_augmented = x_train[randint].copy()
y_augmented = y_train[randint].copy()

x_augmented = datagen2.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle = False
).next()[0]

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


np.save(path_save +'keras58_7_rsp_x_train.npy', arr = x_train)
np.save(path_save +'keras58_7_rsp_x_test.npy', arr = x_test)
np.save(path_save +'keras58_7_rsp_y_train.npy', arr = y_train)
np.save(path_save +'keras58_7_rsp_y_test.npy', arr = y_test)
 
from image import rembg