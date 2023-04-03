import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#이미지 전처리
train_datagen = ImageDataGenerator(           #치환과 증폭 가능 원핫인코딩도 제공
    rescale=1./255,         #'.':부동소수점으로 연산해라  #MinMaxscaler:정규화(Nomalization)=이미지: /255
    horizontal_flip=True,   #수평을 반전(상하반전)  #주의) 숫자 6,9있을 경우 사용하면 안됨
    vertical_flip=True,     #좌우반전
    width_shift_range=0.1,  #10%만큼 좌우로 이동 할 수 있다
    height_shift_range=0.1, #데이터 증폭 : 10% 상하이동
    rotation_range=5,       #지정된 각도 범위내에서 임의로 원본이미지를 회전
    zoom_range=1.2,         #약 20%까지 확대
    shear_range=0.7,        #찌그러트림(밀림 강도)
    fill_mode='nearest',    #10% 이동한 이후 생기는 빈자리를 근접값(비슷한 수치)로 채움
) #이러한 옵션(데이터 증폭옵션)들 중 선택해서 이미지 전처리 가능 (훈련데이터) 

test_datagen = ImageDataGenerator(
    rescale=1./255,
) #평가데이터를 증폭한다는 것은 값을 조작한다는 의미=> 따라서, 통상적으로 testdata는 전처리x(증폭x) 


#D드라이브에서 데이터 가져오기 
xy_train = train_datagen.flow_from_directory( #이미지제너레이터는 폴더별로 라벨값 부여
    'd:/study_data/_data/brain/train/', #분류된 폴더의 상위폴더까지 지정  #directory=폴더
    target_size=(200,200),        #수집한 데이터마다 이미지 사진크기 다르므로 이미지크기 동일하게 고정
    batch_size=10, 
    class_mode='binary', #0,1로 구별(nomal,ad) / 0,1,2(가위,바위,보)// #원핫사용한 경우 => 'categorical'
    color_mode='grayscale',
     # color_mode='rgb', #컬러 (5, 200, 200, 3)  #cf) rgba :투명도  (5, 200, 200, 4)
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/', #분류된 폴더의 상위폴더까지 지정
    target_size=(200,200),        #수집한 데이터마다 이미지 사진크기 다르므로 이미지크기 동일하게 고정
    batch_size=5, 
    class_mode='binary', #0,1로 구별(nomal,ad) / 0,1,2(가위,바위,보)// #원핫사용한 경우 => 'categorical'
    color_mode='grayscale',
    shuffle=True,
)

'''
#수치화 된 것 확인가능(실행) #수치화해서 iterator형태로 구성함 
Found 160 images belonging to 2 classes. #xy_train #x_train.shape = (160,200,200,1)            #y_train.shape =(160,)
Found 120 images belonging to 2 classes. #xy_test  #x_test.shape = (120,200,200,1)로 바뀜(흑백) #y_test.shape  =(120,) : 이미지제너레이터는 폴더별로 라벨값 부여하므로 y는 (0,1)->120
# 카데고리별 개수 확인
# pandas - value_counts / numpy - unique 
'''

print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x0000028A4F535F70>
print(xy_train[0])
'''
(x[0] : array([[[[0.22352943],
         [0.22352943],
         [0.22352943],
         ...,
         [[0.        ],
         [0.        ],
         [0.        ],
         ...,
         [0.        ],
         [0.        ],
         [0.        ]]]]
y[0] : array([0., 1., 1., 1., 1.]
'''

# print(xy_train.shape) #error => #numpy, pandas만          ## 리스트나  이터레이터에 쉐이프가 없다 
print(len(xy_train))        # 32 [(160/5=32), (batch_size로 잘려져있음)]/ [0]~[31]까지 있음/ [0][0]=x, [0][1]=y
print(len(xy_train[0]))     # 2  (x,y)/ 첫번째 batch
print(xy_train[0][0])       # x : 5개 들어가있음 (batch=5일때)
print(xy_train[0][1])       # y : [0. 1. 1. 1. 0.]
print(xy_train[0][0].shape) #(5, 200, 200, 1)  #numpy형태라 shape가능
print(xy_train[0][1].shape) #(5,)

'''
x와 y가 합쳐진 iterator형태는 같이 넣어도 됨
 *batch_size=10일 때  
print(xy_train[0][0])        #x 10개 들어가있음
print(xy_train[0][1])        #[0. 1. 0. 1. 1. 0. 1. 1. 1. 0.]
print(xy_train[0][0].shape)  #(10, 200, 200, 1)
print(xy_train[0][1].shape)  #(10,)
'''

print("===========================================================")
print(type(xy_train))  #<class 'keras.preprocessing.image.DirectoryIterator'> : Iterator형태로 구성되어있음 
print(type(xy_train[0])) #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

###x와 y가 합쳐진 iterator형태이다!!###