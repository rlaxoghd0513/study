from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten


model = Sequential()
model.add(Conv2D(7, (2,2),
                 padding= 'valid',           #패딩 적용 되는거    커널 사이즈가 바뀌어도 행,렬은 똑같이 유지될수 있게 이미지가 커진다
                 input_shape = (8,8,1)
                 ,strides = 1))    #출력 (N,8,8,7)     패딩된 이미지의 값들은 0으로 원래 이미지값에 영향을 끼치지 않는다
model.add(Conv2D(filters = 4, 
                 padding = 'valid',  #  패딩의 디폴트 valid 패딩 적용 안되는거       
                 kernel_size = (4,4),     
                 activation='relu',
                 strides=2))       #출력 (N,6,6,4) 
                                           
model.add(Conv2D(10, (2,2),padding = 'same')) #출력 (N,6,6,10)
model.add(Flatten())    
model.add(Dense(32, activation='relu'))  
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))
model.summary()
#stride 디폴트 1 maxpooling 디폴트 2 커널사이즈보다 작게 준다 stride 보폭 커널사이즈의 보폭
#홀수 크기는 맥스풀링하면 남는 가장자리 데이터는 날라간다 
#스트라이드 했는데 남는 데이터가 없으면 포함된 데이터도 버린다