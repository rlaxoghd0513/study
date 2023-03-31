from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(rescale=1.255,
                                   horizontal_flip = True,
                                   vertical_flip= True,
                                   width_shift_range=0.1,
                                   height_shift_range = 0.1,
                                   rotation_range = 5,
                                   zoom_range = 0.1,
                                   shear_range = 0.7,
                                   fill_mode='nearest')

#증폭(증강) 
augment_size = 100


print(x_train.shape)    #(60000, 28, 28)
print(x_train[0].shape) #(28, 28) #x_train의 0번째 쉐잎
print(x_train[1].shape) #(28, 28) #x_train의 1번째 쉐잎
print(x_train[2].shape) #(28, 28) #x_train의 2번째 쉐잎
print(x_train[0][0].shape) #(28,) 
print(x_train[0][1].shape) #(28,)

print(np.tile(x_train[0].reshape(28*28), #수치증폭하려고 리쉐잎 
              augment_size).reshape(-1,28,28,1).shape) #(100, 28, 28, 1) #다시 원래 모양으로 리쉐잎

#np.tile(데이터, 증폭시킬개수)


print(np.zeros(augment_size)) #100개의 0을 출력해준다
print(np.zeros(augment_size).shape) #(100,)

x_data = train_datagen.flow(   #플로우디렉토리는 경로에있는걸 받아들인다  플로우는 데이터를 받아들인다
    np.tile(x_train[0].reshape(28*28), 
              augment_size).reshape(-1,28,28,1),  #x데이터
    np.zeros(augment_size), #y데이터 그림만 그릴거라 필요없어서 걍 0 100개로 함
    batch_size = augment_size,
    shuffle=True
)

print(x_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x0000014F9ED05F10>
print(x_data[0])  #x와 y가 모두 포함
print(x_data[0][0].shape) #(100,28,28,1)
print(x_data[0][1].shape) #(100,)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)  #49개의 플롯을 그린다
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap = 'gray') #하나씩
plt.show()
    





