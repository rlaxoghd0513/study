import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
#1.데이터
x = np.array([1,2,3,4,5])
x = np.array([1,2,3,4,5])

#2.모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

print(model.weights)
print("############################################################")
print(model.trainable_weights)
print("############################################################")
print(len(model.weights))
print(len(model.trainable_weights))

#################################중요#####################################
model.trainable = False 
##########################################################################

print(len(model.weights))
print(len(model.trainable_weights))

model.summary()