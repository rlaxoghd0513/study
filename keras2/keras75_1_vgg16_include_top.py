#전이학습
# 와꾸 달라도 기울기는 잘 맞는다 사전학습 따로 훈련시키지 않는다
# 입력과 출력만 custom해주면 된다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

#model = VGG16()  #include_top = (224,224,3) 디폴트
model = VGG16(weights = 'imagenet',
              include_top = False,#True로 하면 (224,224,3) 이 인풋이여야 한다
              input_shape = (32,32,3)
              )

model.summary()

print(len(model.weights)) #32 에서 26로 변한다
print(len(model.trainable_weights)) #32 에서 26으로 변한다

##########################################################################################
# include_top=True일 경우
# FC layer 원래꺼 쓴다 (fully connected layer)
# input이 (224,224,3) 이여야한다

#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0
# ....

#  flatten (Flatten)           (None, 25088)             0

#  fc1 (Dense)                 (None, 4096)              102764544

#  fc2 (Dense)                 (None, 4096)              16781312

#  predictions (Dense)         (None, 1000)              4097000

# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0

#############################################################################################
# include_top=False일 경우
# FC layer 원래꺼 삭제됨 -> 커스텀마이징한다
# input은 데이터에 맞게 바꿀수 있다

#  input_1 (InputLayer)        [(None, 32, 32, 3)]       0

#  block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________