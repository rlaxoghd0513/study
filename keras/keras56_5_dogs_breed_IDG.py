import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

path = 'd:/study_data/_data/dogs_breed/'
path_save = 'd:/study_data/_save/dogs_breed/'

datagen = ImageDataGenerator(rescale = 1./255)

db = datagen.flow_from_directory(
    path,
    target_size = (200,200),
    batch_size = 200,
    class_mode = 'categorical',
    color_mode= 'rgb',
    shuffle = True
)

db_x = db[0][0]
db_y = db[0][1]

db_x_train , db_x_test, db_y_train, db_y_test = train_test_split(db_x, db_y,
                                                                 random_state = 333,
                                                                 shuffle = True,
                                                                 train_size = 0.7)

print(db_x_train.shape , db_x_test.shape)
print(db_y_train.shape, db_y_test.shape)

np.save(path_save + 'keras56_db_x_train.npy', arr = db_x_train)
np.save(path_save + 'keras56_db_x_test.npy', arr = db_x_test)
np.save(path_save + 'keras56_db_y_train.npy', arr = db_y_train)
np.save(path_save + 'keras56_db_y_test.npy', arr = db_y_test)
