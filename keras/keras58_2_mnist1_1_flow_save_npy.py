from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   width_shift_range = 0.1,
                                   zoom_range = 0.1,
                                   fill_mode = 'nearest')

train_datagen2 = ImageDataGenerator(rescale = 1./255)

augment_size = 40000

np.random.seed(42)
randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2],
                        1) 

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],
                                  1)


x_augmented = train_datagen.flow(#튜플의 이터레이터 형태로 만드는거 flow
    x_augmented,y_augmented,
    batch_size = augment_size,
    shuffle = False,
).next()[0]

print(np.max(x_train), np.min(x_train)) #255 0
print(np.max(x_augmented), np.min(x_augmented))#1.0 0.0

x_train =np.concatenate((x_train/255.,x_augmented))
y_train = np.concatenate((y_train, y_augmented))
x_test = x_test/255.
print(x_train.shape, y_train.shape)

path_save = 'd:/study_data/_save/mnist/'

np.save(path_save + 'keras58_2_mnist_flow_x_train.npy', arr = x_train)
np.save(path_save + 'keras58_2_mnist_flow_x_test.npy', arr = x_test)
np.save(path_save + 'keras58_2_mnist_flow_y_train.npy', arr = y_train)
np.save(path_save + 'keras58_2_mnist_flow_y_test.npy', arr = y_test)
