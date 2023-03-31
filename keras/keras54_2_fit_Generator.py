import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터
#이미지 전처리
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
    batch_size= 5,                    ###전체 데이터 쓸려면 160넣기(통배치)###
    class_mode='binary', 
    color_mode='grayscale',
    # color_mode='rgb', #컬러 (5, 200, 200, 3)  #cf) rgba :투명도  (5, 200, 200, 4)
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',
    target_size=(100,100),       
    batch_size=5, 
    class_mode='binary', 
    color_mode='grayscale', 
    shuffle=True,
)

print(xy_train)    # <keras.preprocessing.image.DirectoryIterator object at 0x0000028A4F535F70>
print(xy_train[0])

print(len(xy_train))        # 32 [(160/5=32), (batch_size로 잘려져있음)]/ [0]~[31]까지 있음/ [0][0]=x, [0][1]=y
print(len(xy_train[0]))     # 2  (x,y)/ 첫번째 batch
print(xy_train[0][0])       # x : 5개 들어가있음 (batch=5일때)
print(xy_train[0][1])       # y : [0. 1. 1. 1. 0.]
print(xy_train[0][0].shape) #(5, 200, 200, 1)  #numpy형태라 shape가능
print(xy_train[0][1].shape) #(5,)

#현재 x는 (5,200,200,1) 짜리 데이터가 32덩어리 


#2. 모델 구성 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(100,100,1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
'''
# model.fit(xy_train[:][0], xy_train[:][1], epochs=10)  #TypeError: '>=' not supported between instances of 'slice' and 'int'
# model.fit(xy_train[0][0],xy_train[0][1], epochs=10)     #int형태이므로 돌어감
#이미지를 가져와서 batch하는것이 아니라, batch하고 이미지가져와서 한번에 처리하겠다. 
###전체 데이터 쓸려면 160넣기(통배치)= 데이터를 통으로 쓰고싶다하면 batch_size를 전체 데이터 개수(160)이상으로 잡으면 됨### 
(160,100,100,1)(160,) =>[0][0],[0][1]하면 x,y모두 가져오는 것임 (통배치일 경우[1][0],[1][0]은 없으므로..)
'''
hist = model.fit_generator(xy_train, epochs=100,  # (fit_generator) x데이터,y데이터,batch_size까지 된 것
                    steps_per_epoch=32,   # 훈련(train)데이터/batch = 160/5=32 (32가 한계사이즈임(max), 이만큼 잡아주는게 좋음/이상 쓰면 과적합, 더 적은 숫자일 경우 훈련 덜 돌게 됨)
                    validation_data=xy_test,
                    validation_steps=24,  # val(test)데이터/batch = 120/5=24
                    )   
#history=(metrics)loss, val_loss, acc
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print(acc) 
print("acc:",acc[-1])
print("val_acc:",val_acc[-1])
print("loss:",loss[-1])
print("val_loss:",val_loss[-1])


#[실습1]그림그리기 subplot(두개 그림을 하나로)
#[실습] 튜닝 acc 0.95이상


#그림(그래프)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

# plt.subplot(nrows,ncols,index)
# plt.subplot(총 행 개수, 총 열 개수, 그래프 번호)
plt.subplot(1,2,1)
plt.title('Loss')
plt.plot(hist.history['loss'],marker='.', label='loss', c='red')
plt.plot(hist.history['val_loss'], marker='.', label='val_loss', c='blue')
plt.legend() #범례표시
# plt.xlabel('epochs')
# plt.ylabel('loss, val_loss')
# plt.grid() #격자표시

plt.subplot(1,2,2)
plt.title('Acc')
plt.plot(hist.history['acc'],  marker='.', label= 'acc', c='red')
plt.plot(hist.history['val_acc'], marker='.', label='val_acc', c='blue')
plt.legend()

plt.show()