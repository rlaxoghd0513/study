import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,         
    horizontal_flip=True,   
    vertical_flip=True,    
    width_shift_range=0.1,  
    height_shift_range=0.1, 
    rotation_range=5,       
    zoom_range=1.2,        
    shear_range=0.7,        
    fill_mode='nearest',
    ) 

test_datagen = ImageDataGenerator(rescale=1./255,) 


#D드라이브에서 데이터 가져오기 
xy_train = train_datagen.flow_from_directory( 
    'd:/study_data/_data/brain/train/',
    target_size=(100,100),       
    batch_size= 160,                   
    class_mode='binary',   
    color_mode='grayscale',
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',
    target_size=(100,100),       
    batch_size= 120, 
    class_mode='binary', 
    color_mode='grayscale', 
    shuffle=True,
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, x_test.shape)#(160, 100, 100, 1)(120, 100, 100, 1)
print(y_train.shape, y_test.shape)#(160,) (120,)

augment_size = 140

np.random.seed(42)
randint = np.random.randint(160, size = 140)
print(randint.shape)

x_augmented = x_train[randint].copy()
y_augmented = y_train[randint].copy()

print(type(x_train))#<class 'numpy.ndarray'>
print(type(x_augmented))#<class 'numpy.ndarray'>
print(x_augmented.shape) #(140, 100, 100, 1)


#타입맞춰줄라하는게 아니라 증폭시킨 데이터니까 이미지데이터제너레이터 써서 변환 시킬라고
x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle = False
).next()[0]

x_train = np.concatenate((x_train/255., x_augmented))
y_train = np.concatenate((y_train/255., y_augmented))
x_test = x_test/255.


path ='d:/study_data/_save/brain/'

np.save(path + 'keras58_5_brain_x_train.npy', arr = x_train)
np.save(path + 'keras58_5_brain_x_test.npy', arr = x_test)
np.save(path + 'keras58_5_brain_y_train.npy', arr = y_train)
np.save(path + 'keras58_5_brain_y_test.npy', arr = y_test)