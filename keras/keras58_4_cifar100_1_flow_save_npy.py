from tensorflow.keras.datasets import cifar100
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

(x_train,y_train), (x_test, y_test)= cifar100.load_data()

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   horizontal_flip= True,
                                   height_shift_range=0.1,
                                   fill_mode = 'nearest')

train_datagen = ImageDataGenerator(rescale=1./1)

print(x_train.shape, x_test.shape) #(50000, 32, 32, 3)(10000, 32, 32, 3)

augment_size = 50000

np.random.seed(42)
randint = np.random.randint(x_train.shape[0], size = augment_size)
print(randint.shape) #(50000,)

x_augmented = x_train[randint].copy()
y_augmented = y_train[randint].copy()

print(x_train.shape) #(50000, 32, 32, 3)
print(x_test.shape) #(10000, 32, 32, 3)
print(x_augmented.shape) #(50000, 32, 32, 3)

print(type(x_train)) #<class 'numpy.ndarray'>
print(type(x_augmented)) #<class 'numpy.ndarray'>

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle = False,
).next()[0]

print(x_augmented.shape) #(50000, 32, 32, 3)

x_train = np.concatenate((x_train/255., x_augmented))
y_train = np.concatenate((y_train, y_augmented))

x_test = x_test/255.

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


path_save = 'd:/study_data/_save/cifar100/'
np.save(path_save + 'keras58_4_cifar100_flow_x_train.npy', arr = x_train)
np.save(path_save + 'keras58_4_cifar100_flow_x_test.npy', arr = x_test)
np.save(path_save + 'keras58_4_cifar100_flow_y_train.npy', arr = y_train)
np.save(path_save + 'keras58_4_cifar100_flow_y_test.npy', arr = y_test)