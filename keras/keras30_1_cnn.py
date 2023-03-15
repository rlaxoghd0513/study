from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten



model = Sequential()
model.add(Conv2D(7, (2,2),               #이미지는 행렬rgb(채널)장수 4차원
                 input_shape = (8,8,1))) #가로가 8 세로가 8 1은 칼라 흑백  (2,2)는 자르는 크기 (7,7)로 바뀜  높은놈은 높아지고 낮은놈은 낮아지고 그 데이터를 7장으로 늘렸다
                                         #특성은 좁히고 데이터는 늘린다   통과했을때 (4.4.7)이 된다  #출력4차원   : (N,7,7,7) 
                                                                                                 # 인풋사이즈의 정식명칭 #(batch_size, lows, columns, channels) 칼라면 3 흑백이면 1 
                                                                                                #  channels는  레이어 한번 거치면 필터로 바뀜          
model.add(Conv2D(filters = 4,            
                 kernel_size = (3,3),      #얼마의 크기로 자를지 
                 activation='relu'))       #(2,2)가 4장으로 바뀌고 4차원 안에 모든 값들은 relu를 거쳐 음수부분은 0이 된다 출력(N, 5,5,4)

model.add(Conv2D(10, (2,2)))  #출력(N, 4,4,10)

#4차원을 쭉 핀다
model.add(Flatten()) #상단에 있는 레이어 평탄화     #출력(N,4*4*10)-> (N,160)
model.add(Dense(32, activation='relu'))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))
model.summary()







