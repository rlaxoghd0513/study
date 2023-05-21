#pip install keras==1.2.2
# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
import time
# import keras
if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
# print(keras.__version__) #Using TensorFlow backend. 텐서플로를 바닥에 깔고 시작. 그래서 느림.

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

#[실습] 맹그러

# One-hot 인코딩
encoder = OneHotEncoder(sparse=False)
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, y_train.shape) #(60000, 784) (60000, 10)
# print(x_test.shape, y_test.shape)   #(10000, 784) (10000, 10)

# 2. 모델
#레이어1 CNN
x = tf.compat.v1.placeholder('float', shape=[None, 28, 28, 1])
y = tf.compat.v1.placeholder('float', shape=[None, 10])
                                      # kernal_size = (3,3), channels, filters
w1 = tf.compat.v1.Variable(tf.random.normal(shape=[3, 3, 1, 64]))
    # model.add(Conv2D(32,kernal_size=(3,3))), input_shape = (28,28,1)
b1 = tf.compat.v1.Variable(tf.zeros([64]), name='bias1')
layer1 = tf.compat.v1.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
layer1 += b1
L1_maxpool = tf.nn.max_pool2d(layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# 컨볼루션 레이어 하나.(36~41줄) -> (n, 14, 14, 64)

dropout1 = tf.compat.v1.nn.dropout(layer1, rate = 0.3) #드랍아웃
#레이어2 CNN
w2 = tf.compat.v1.Variable(tf.random.normal(shape=[3, 3, 64, 32]))
b2 = tf.compat.v1.Variable(tf.zeros([32]), name='bias2')
layer2 = tf.compat.v1.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='VALID')
layer2 += b2
L2_maxpool = tf.nn.max_pool2d(layer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 전에나오던것에 반띵임.
#(n, 12, 12, 32) -> (n, 6, 6, 32)

#레이어3 CNN
w3 = tf.compat.v1.Variable(tf.random.normal(shape =[3, 3, 32, 16]))
b3 = tf.compat.v1.Variable(tf.zeros([16]), name='bias3')
layer3 = tf.compat.v1.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
layer3 += b3 #(n, 6, 6, 16)

#Flatten
L_flat = tf.reshape(layer3, [-1, 6*6*16])

##레이어 4 Dnn
w4 = tf.compat.v1.Variable(tf.random.normal(shape =[6*6*16, 100]))
b4 = tf.compat.v1.Variable(tf.zeros([100]), name='bias4')
layer4 = tf.nn.relu((tf.compat.v1.matmul(L_flat, w4) + b4))
layer4 = tf.nn.dropout(layer4, rate=0.3)

##레이어 5 Dnn Output
w5 = tf.compat.v1.Variable(tf.random.normal(shape=[576, 10]))
b5 = tf.compat.v1.Variable(tf.zeros([10]), name='bias5')
hypothesis = tf.compat.v1.matmul(L_flat, w5) + b5
hypothesis = tf.nn.softmax(hypothesis)

# 3. 컴파일, 훈련
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=hypothesis))
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis = 1))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

# 3. FIT
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 100

epochs = 100
total_batch = int(len(x_train)/batch_size) # 60000/100 = 600
start_time = time.time()

avg_cost = 0

for step in range(epochs):
    avg_cost = 0
    for i in range(int(total_batch)):
        start = i * batch_size
        end = start + batch_size
        
        cost_val, _ = sess.run([loss, train], feed_dict={x: x_train[start:end], y: y_train[start:end]})
        avg_cost += cost_val / total_batch
    print("EPOCHS:", step + 1, 'LOSS: {:.9f}'.format(avg_cost))

end_time = time.time()
print("DONE")

# 4. PREDICT
y_predict = sess.run(tf.argmax(hypothesis, axis=1), feed_dict={x:x_test})
y_predict_arg = sess.run(tf.argmax(y_predict, 1))

y_test_arg = np.argmax(y_test,1)
 
acc = accuracy_score(y_predict_arg, y_test_arg)
print("accuracy_score:", acc)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=np.float32))

acc = sess.run([hypothesis, predicted, accuracy],
                    feed_dict={x:x_test, y:y_test})

print("ACC: ",  acc)
print("TIME: ", end_time - start_time)

# # 4. 모델 훈련
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# batch_size = 100
# total_batch = int(len(x_train/batch_size)) # 60000/100 = 600
# epochs = 1000

# import time
# start = time.time()
# for epoch in range(epochs):
#     avg_loss=0
    
#     for i in range(int(total_batch)): #100개씩 600번 돌기/
#         start = i * batch_size        # 0, 10, 200, ...   59900
#         end = start + batch_size      # 100,200, 300, ... 60000
    
#     _, loss_val = sess.run([train, loss], 
#                             feed_dict={x: x_train[start:end], y: y_train[start:end]})

#     avg_loss += loss_val/ total_batch
#     # -> one epochs
#     print("Epochs:", epoch + 1, "Loss: {:.9f}".format(avg_loss))    

# print("훈련끝")

# # 훈련된 모델을 통해 예측값 출력
# y_pred = sess.run(hypothesis, feed_dict={x: x_test})
# # print("Predictions:", y_pred)

# # 평가 지표 계산
# y_pred_label = np.argmax(y_pred, axis=1)
# y_test_label = np.argmax(y_test, axis=1)

# acc = accuracy_score(y_test_label, y_pred_label)
# print("Accuracy:", acc)

# sess.close()
# # # Accuracy: 0.8625