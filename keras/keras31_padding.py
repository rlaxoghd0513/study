from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten



# model = Sequential()
# model.add(Conv2D(7, (2,2),
#                  padding= 'same',           #패딩 적용 되는거
#                  input_shape = (8,8,1)))    #출력 (N,8,8,7)
# model.add(Conv2D(filters = 4, 
#                  padding = 'valid',  #  패딩의 디폴트 valid 패딩 적용 안되는거       
#                  kernel_size = (3,3),     
#                  activation='relu'))       #출력 (N,6,6,4)
                                           
# model.add(Conv2D(10, (2,2),padding = 'same')) #출력 (N,6,6,10)
# model.add(Flatten())    
# model.add(Dense(32, activation='relu'))  
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))
# model.summary()


model = Sequential()
model.add(Conv2D(7, (3,3),
                 padding= 'same',           #패딩 적용 되는거    커널 사이즈가 바뀌어도 행,렬은 똑같이 유지될수 있게 이미지가 커진다
                 input_shape = (8,8,1)))    #출력 (N,8,8,7)     패딩된 이미지의 값들은 0으로 원래 이미지값에 영향을 끼치지 않는다
model.add(Conv2D(filters = 4, 
                 padding = 'valid',  #  패딩의 디폴트 valid 패딩 적용 안되는거       
                 kernel_size = (3,3),     
                 activation='relu'))       #출력 (N,6,6,4) 
                                           
model.add(Conv2D(10, (2,2),padding = 'same')) #출력 (N,6,6,10)
model.add(Flatten())    
model.add(Dense(32, activation='relu'))  
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))
model.summary()