import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from sklearn.model_selection import train_test_split

path = 'd:/study_data/_data/horse_or_human/'
path_save = 'd:/study_data/_save/horse_or_human/'

datagen = ImageDataGenerator(rescale=1./255)
hoh = datagen.flow_from_directory(path, target_size = (150,150), batch_size = 100, class_mode = 'binary', color_mode = 'rgb')

hoh_x = hoh[0][0]
hoh_y = hoh[0][1]

hoh_x_train, hoh_x_test, hoh_y_train, hoh_y_test = train_test_split(hoh_x, hoh_y, train_size = 0.7, random_state = 333, shuffle=True)

np.save(path_save + 'keras56_hoh_x_train.npy', arr=hoh_x_train)
np.save(path_save + 'keras56_hoh_x_test.npy', arr = hoh_x_test)
np.save(path_save + 'keras56_hoh_y_train.npy', arr = hoh_y_train)
np.save(path_save + 'keras56_hoh_y_test.npy', arr = hoh_y_test)
