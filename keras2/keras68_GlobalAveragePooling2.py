from tensorflow.keras.datasets import cifar100
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import GlobalAveragePooling2D

import tensorflow as tf
tf.random.set_seed(337) #난수고정

#1 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)

x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# x_train = x_train/255.  #이미지에서는 어차피 rgb가 0부터 255까지 니까 255로 나눠도 된다
# x_test = x_test/255.

# x_train = x_train.reshape(50000, 28,28,1)/255.
# x_test = x_test.reshape(10000, 28,28,1)/255.       스케일러 대신  이렇게도 된다

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

##################실습#####################
x_train = x_train.reshape(50000,32,32,3)   #데이터는 바뀐게 없고 순서도 바뀐게 없다  reshape는 구조만 바뀌는거지 순서와 내용은 바뀌지 않는다 (28,14,2)도 가능하다  
                                          # 아파트로 생각하면 28층짜리 28개나 28층짜리 14동 2개나 같다
x_test = x_test.reshape(10000,32,32,3)     #transpose는 행과 열이 바뀐다  

print(np.unique(y_train, return_counts = True)) 



from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
print(y_train.shape)
y_test = to_categorical(y_test)
print(y_test.shape)

#2 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', input_shape=(32,32,3)))
model.add(MaxPooling2D())    #커널처럼 중첩되지 않고 각 구역에서 가장 큰 애만 뽑는다
model.add(Conv2D(64, (2,2), padding='valid', activation='relu'))
model.add(Conv2D(32, 2)) #대부분 2,2 3,3으로 커널사이즈를 하니까 귀찮아서 2,2를 2만 써도 된다
model.add(Flatten())
# model.add(GlobalAveragePooling2D()) #Flatten대신에 쓸수 있다 , Flatten보다 통상 괜찮다 
model.add(Dense(32))
model.add(Dense(100, activation = 'softmax'))
model.summary()

#Flatten이랑 GlobalAveragePooling2D 연산량 비교
############################ Flatten ######################################




################## GlobalAveragePooling2D #########################
# conv2d_2 (Conv2D)            (None, 14, 14, 32)        8224
# _________________________________________________________________
# module_wrapper (ModuleWrappe (None, 32)                0
# _________________________________________________________________
# dense (Dense)                (None, 32)                1056
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               3300
# =================================================================
# Total params: 29,860
# Trainable params: 29,860
# Non-trainable params: 0

'''
#3 컴파일 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_acc', mode = 'max', patience=20, verbose=1, restore_best_weights=True)
model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_split=0.2, callbacks = [es])

#4 평가 예측
results = model.evaluate(x_test, y_test)
print('results:', results)

y_predict = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)
y_predict_acc = np.argmax(y_predict, axis=1)

acc=accuracy_score(y_test_acc, y_predict_acc)
print('acc:',acc)
'''