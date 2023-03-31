import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from sklearn.model_selection import train_test_split

path = 'd:/study_data/_data/rsp/'
path_save = 'd:/study_data/_save/rsp/'

datagen = ImageDataGenerator(rescale = 1./255)
rsp = datagen.flow_from_directory(path, target_size = (100,100), batch_size = 100, class_mode = 'categorical', color_mode = 'rgb', shuffle = True)

rsp_x = rsp[0][0]
rsp_y = rsp[0][1]

rsp_x_train, rsp_x_test, rsp_y_train, rsp_y_test = train_test_split(rsp_x, rsp_y, train_size = 0.725, random_state = 333, shuffle=True)

print(rsp_x_train.shape, rsp_x_test.shape)#(70, 200, 200, 3) (30, 200, 200, 3)    #하위폴더에 있는 폴더 갯수만큼 카테고리컬 된다
print(rsp_y_train.shape, rsp_y_test.shape)#(72, 3) (28, 3)     


np.save(path_save +'keras56_rsp_x_train.npy', arr = rsp_x_train)
np.save(path_save +'keras56_rsp_x_test.npy', arr = rsp_x_test)
np.save(path_save +'keras56_rsp_y_train.npy', arr = rsp_y_train)
np.save(path_save +'keras56_rsp_y_test.npy', arr = rsp_y_test)

print(rsp_x_train.shape, rsp_x_test.shape)
print(rsp_y_train.shape, rsp_y_test.shape)