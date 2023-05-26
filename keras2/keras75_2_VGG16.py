import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights = 'imagenet',
              include_top = False,#True로 하면 (224,224,3) 이 인풋이여야 한다
              input_shape = (32,32,3)
              )

vgg16.trainable = False    #vgg의 가중치는 변하지 않는다

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation = 'softmax'))

# model.trainable = True #디폴트 True

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

# model.Trainable = True일때 
# 30
# 30
# model.Trainable = False일때
# 30
# 0
# vgg16.trainable = False
# 30
# 4

#통상적으로 vgg16을 동결시키는게 좋긴하지만 케바케다 동결안시키는게 좋을 수도 있음
