import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터
#이미지 전처리       #정의
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
    batch_size= 5000,                    ###전체 데이터 쓸려면 160넣기(통배치)###
    class_mode='binary',        # y라벨 개수 클래스가 뭐뭐 있냐
    color_mode='grayscale',
    # color_mode='rgb', #컬러 (5, 100, 100, 3)  #cf) rgba :투명도  (5, 100, 100, 4)
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',
    target_size=(100,100),       
    batch_size=1005, 
    class_mode='binary', 
    color_mode='grayscale', 
    shuffle=True,
)

print(xy_train)
print(xy_train[0])
print(len(xy_train))
print(len(xy_test))

print(xy_train[0][0].shape)
print(xy_test[0][0].shape)
print(xy_train[0][1].shape)
print(xy_test[0][1].shape)


path ='d:/study_data/_save/_npy/'

np.save(path + 'keras55_1_x_train.npy', arr = xy_train[0][0])
np.save(path + 'keras55_1_x_test.npy', arr = xy_test[0][0])
np.save(path + 'keras55_1_y_train.npy', arr = xy_train[0][1])
np.save(path + 'keras55_1_y_test.npy', arr = xy_test[0][1])











'''
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
model.add(Dense(2, activation='sigmoid'))


#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

#1)통배치로 fit
# hist = model.fit(xy_train[0][0], xy_train[0][1], batch_size=16, epochs=10,
#           validation_data=(xy_test[0][0],xy_test[0][1]))  

#2)fit_generator
# hist = model.fit_generator(xy_train, epochs=100,  # (fit_generator) x데이터,y데이터,batch_size까지 된 것
#                     steps_per_epoch=32,   # 훈련(train)데이터/batch = 160/5=32 (32가 한계사이즈임(max), 이만큼 잡아주는게 좋음/이상 쓰면 과적합, 더 적은 숫자일 경우 훈련 덜 돌게 됨)
#                     validation_data=xy_test,
#                     validation_steps=24,  # val(test)데이터/batch = 120/5=24
#                     )   

#3)fit
hist = model.fit(xy_train, epochs=1000,  # (fit_generator) x데이터,y데이터,batch_size까지 된 것
                    steps_per_epoch=10,   # 훈련(train)데이터/batch = 160/5=32 (32가 한계사이즈임(max), 이만큼 잡아주는게 좋음/이상 쓰면 과적합, 더 적은 숫자일 경우 훈련 덜 돌게 됨)
                    validation_data=xy_test,
                    validation_steps=120/16,  # val(test)데이터/batch = 120/5=24
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
'''
