#자기사진에 노이즈 집어넣고 수정

import numpy as np
path = 'd:/study/_data/'
path_save = 'd:/study/_save/'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

datagen = ImageDataGenerator(rescale=1./255)

mw = datagen.flow_from_directory(
    path,
    target_size = (200,200),
    batch_size = 32,
    class_mode = 'input',
    color_mode = 'rgb',
    shuffle = True
)

mw_x = mw[0][0]
mw_y = mw[0][1]

print(mw_x.shape)
print(mw_y.shape)



np.save(path_save + '증사_x.npy', arr = mw_x)
np.save(path_save + '증사_y.npy', arr = mw_y)
