import numpy as np#2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from sklearn.model_selection import train_test_split

path = 'd:/study_data/_data/men_women/'
path_save = 'd:/study_data/_save/men_women_flow/'

datagen = ImageDataGenerator(rescale = 1./255)
datagen2 = ImageDataGenerator(horizontal_flip=True,
                              height_shift_range=0.1,
                              fill_mode= 'nearest')
rsp = datagen.flow_from_directory(path, target_size = (100,100), batch_size = 100, class_mode = 'binary', color_mode = 'rgb', shuffle = True)

x = rsp[0][0]
y = rsp[0][1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state = 333, shuffle=True)

print(x_train.shape, x_test.shape)#   (70, 100, 100, 3) (30, 100, 100, 3) #하위폴더에 있는 폴더 갯수만큼 카테고리컬 된다
print(y_train.shape, y_test.shape)#(70,) (30,)
    
print(np.min(x_train), np.max(x_train))#0.0 1.0
print(np.min(x_test), np.max(x_test))#0.0 1.0

augment_size = 50
np.random.seed(42)
randint = np.random.randint(70,size=50)
x_augmented = x_train[randint].copy()
y_augmented = y_train[randint].copy()

x_augmented = datagen2.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle = False
).next()[0]

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(np.min(x_train), np.max(x_train)) #0.0 1.0
print(np.min(x_test), np.max(x_test))#0.0 1.0


np.save(path_save +'keras58_10_x_train.npy', arr = x_train)
np.save(path_save +'keras58_10_x_test.npy', arr = x_test)
np.save(path_save +'keras58_10_y_train.npy', arr = y_train)
np.save(path_save +'keras58_10_y_test.npy', arr = y_test)