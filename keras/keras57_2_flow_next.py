from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)


train_datagen = ImageDataGenerator(rescale=1.255,
                                   horizontal_flip = True, #좌우
                                #    vertical_flip= True,   #위아래
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
print(x_train[0][0]) # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
print(x_train[0][1].shape) #(28,) 
print(x_train[0][1])  #[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
print(x_train[0][15].shape) #(28,) 
print(x_train[0][15])  #[  0   3   0   0   0   0   0   0   0  62 145 204 228 207 213 221 218 208 211 218 224 223 219 215 224 244 159   0]
print(x_train[0][27].shape) #(28,)
print(x_train[0][27]) #[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

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
).next() #이터레이터형태의 다음 데이터를 가져온다
##################################################################.next 사용
print(x_data) # x와 y 가 합쳐진 데이터 출력
print(type(x_data)) #class 'tuple'
print(x_data[0]) #x데이터
print(x_data[1]) #y데이터
print(x_data[0].shape, x_data[1].shape) #튜플안에 넘파이가 들어가있는 구조  (100, 28, 28, 1) (100,)
print(type(x_data[0])) #<class 'numpy.ndarray'>

###################################################################.next 미사용
# print(x_data) 
#<keras.preprocessing.image.NumpyArrayIterator object at 0x0000014F9ED05F10>
# print(x_data[0])  #x와 y가 모두 포함
# print(x_data[0][0].shape) #(100,28,28,1)
# print(x_data[0][1].shape) #(100,)



import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):               #100개중에 49개만 빼서 본다
    plt.subplot(7,7,i+1)          #49개의 플롯을 그린다
    plt.axis('off')
    # plt.imshow(x_data[0][0][i], cmap = 'gray') # .next() 미사용
    plt.imshow(x_data[0][i], cmap = 'gray') # .next() 사용
plt.show()
