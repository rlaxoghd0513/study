import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

path = 'd:/study_data/_data/men_women/'
path_save = 'd:/study_data/_save/men_women/'

datagen = ImageDataGenerator(rescale=1./255)

mw = datagen.flow_from_directory(
    path,
    target_size = (200,200),
    batch_size = 100,
    class_mode = 'binary',
    color_mode = 'rgb',
    shuffle = True
)

mw_x = mw[0][0]
mw_y = mw[0][1]

mw_x_train, mw_x_test, mw_y_train, mw_y_test = train_test_split(mw_x, mw_y, train_size = 0.7, shuffle=True, random_state = 123)

print(mw_x_train.shape, mw_x_test.shape)#(70, 200, 200, 3) (30, 200, 200, 3)
print(mw_y_train.shape, mw_y_test.shape)#(70,) (30,)

np.save(path_save + 'keras56_mw_x_train.npy', arr = mw_x_train)
np.save(path_save + 'keras56_mw_x_test.npy', arr = mw_x_test)
np.save(path_save + 'keras56_mw_y_train.npy', arr = mw_y_train)
np.save(path_save + 'keras56_mw_y_test.npy', arr = mw_y_test)




