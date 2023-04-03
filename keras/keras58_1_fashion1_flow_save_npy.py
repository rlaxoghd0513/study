#57_5카피해서 복붙함


from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip = True, #좌우
                                   vertical_flip= True,   #위아래
                                   width_shift_range=0.1,
                                   height_shift_range = 0.1,
                                   rotation_range = 5,
                                   zoom_range = 0.1,
                                   shear_range = 0.7,
                                   fill_mode='nearest')

train_datagen2 = ImageDataGenerator(rescale=1./1,)
                               

#증폭(증강) 
augment_size = 40000 #6만개인 패션엠니스트를 10만개로 증폭할거다 그래서 augment_size 4만

#하나의 데이터를 4만개 증폭시ㅋ면 의미없어서 6만개중 4만개 뽑기
# randidx = np.random.randint(60000, size = 40000)
np.random.seed(42) 
randidx = np.random.randint(x_train.shape[0], size = augment_size)  #(60000,28,28) 의 0번째는 60000  이러면 x_train사이즈가 바뀌어도 굳이 다시 명시안해도 된다 
print(randidx) #[32963 17416 41890 ... 13860  4541 50320]
print(randidx.shape) #(40000,)
print(np.min(randidx), np.max(randidx)) #0 59996 /0 59997 / 계속 달라지니까 np시드 고정 2 59998  2 59998 이제 값이 계속 같다

# x_augmented = x_train[randidx] #4만개 (40000,28,28)
# y_augmented = y_train[randidx] #4만개 (40000,)

x_augmented = x_train[randidx].copy()  # 카피 안붙이면 그냥 똑같은 데이터가 두개가 중복된다 그리고 증폭시킬 데이터를 변환시키면 원데이터도 변환된다 copy()를 붙여주면 원데이터 안변함
y_augmented = y_train[randidx].copy()  #
print(x_augmented)
print(x_augmented.shape, y_augmented.shape)#(40000, 28, 28) (40000,)

#4차원으로 변환
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],#갯수를 모를땐 이렇게 써도 된다
                        x_test.shape[1],
                        x_test.shape[2],
                        1) 

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],
                                  1)
                        
#그냥 복사만 해온 데이터니까 변환시켜야한다

# x_augmented = train_datagen.flow(
#     x_augmented,y_augmented,
#     batch_size = augment_size,
#     shuffle = False,
# )
# #통과하면  x와 y가 합쳐져있는 iterator형태가 된다

# print(x_augmented) #<keras.preprocessing.image.NumpyArrayIterator object at 0x0000027358503FA0>

# print(x_augmented[0][0].shape) #(40000, 28, 28, 1)


# x_augmented = train_datagen.flow(
#     x_augmented,y_augmented,
#     batch_size = augment_size,
#     shuffle = False,
# ).next()   #이러면 이터레이터 안의 첫번째 튜플이 나온다 x_augmented[0]이 나옴

x_augmented = train_datagen.flow(#튜플의 이터레이터 형태로 만드는거 flow
    x_augmented,y_augmented,
    batch_size = augment_size,
    shuffle = False,
).next()[0]

print(x_augmented)
print(x_augmented.shape) #(40000, 28, 28, 1)


print(np.max(x_train), np.min(x_train)) #255 0
print(np.max(x_augmented), np.min(x_augmented)) #1.0 0.0

x_train =np.concatenate((x_train/255.,x_augmented))
y_train = np.concatenate((y_train, y_augmented))
x_test = x_test/255.
print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)

print(type(x_train))#<class 'numpy.ndarray'>
print(type(x_test))##<class 'numpy.ndarray'>
print(type(y_train))#<class 'numpy.ndarray'>
print(type(y_test))#<class 'numpy.ndarray'>
#########################################fit generator 쓸라고 x,y 합치기##################################################
# batch_size = 64
# xy_train = train_datagen2.flow(x_train, y_train, batch_size=batch_size, shuffle=True) #또 스케일링 하면 안댐 튜플의 이터레이터 형태로 만들어주기 위해서 datagen2
# print(type(xy_train)) #<class 'keras.preprocessing.image.NumpyArrayIterator'>

path_save = 'd:/study_data/_save/fashion_mnist/'
np.save(path_save + 'keras_58_1_fashion_flow_x_train.npy', arr = x_train)
np.save(path_save + 'keras_58_1_fashion_flow_x_test.npy', arr = x_test)
np.save(path_save + 'keras_58_1_fashion_flow_y_train.npy', arr = y_train)
np.save(path_save + 'keras_58_1_fashion_flow_y_test.npy', arr = y_test)





