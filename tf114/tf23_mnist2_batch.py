#pip install keras==1.2.2
# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
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
x_train = x_train.reshape(60000,28*28).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, y_train.shape) #(60000, 784) (60000, 10)
# print(x_test.shape, y_test.shape)   #(10000, 784) (10000, 10)

# 2. 모델
x = tf.compat.v1.placeholder('float', shape=[None, 784])
y = tf.compat.v1.placeholder('float', shape=[None, 10])

w1 = tf.compat.v1.Variable(tf.random.normal([784, 128], dtype=tf.float32), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([128], dtype=tf.float32), name='bias1')
layer1 = tf.compat.v1.matmul(x, w1) + b1
dropout1 = tf.compat.v1.nn.dropout(layer1, rate = 0.3) #드랍아웃

w2 = tf.compat.v1.Variable(tf.random.normal([128, 64], dtype=tf.float32), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([64], dtype=tf.float32), name='bias2')
layer2 = tf.nn.selu(tf.compat.v1.matmul(dropout1, w2) + b2) # 다음 인풋은 layer1이 아니라 dropout1

w3 = tf.compat.v1.Variable(tf.random.normal([64, 32], dtype=tf.float32), name='weight3')
b3 = tf.compat.v1.Variable(tf.zeros([32], dtype=tf.float32), name='bias3')
layer3 = tf.nn.relu(tf.compat.v1.matmul(layer2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([32, 16], dtype=tf.float32), name='weight4')
b4 = tf.compat.v1.Variable(tf.zeros([16], dtype=tf.float32), name='bias4')
layer4 = tf.nn.softmax(tf.matmul(layer3, w4) + b4)

w5 = tf.compat.v1.Variable(tf.random.normal([16, 10], dtype=tf.float32), name='weight5')
b5 = tf.compat.v1.Variable(tf.zeros([10], dtype=tf.float32), name='bias5')
hypothesis = tf.nn.softmax(tf.matmul(layer4, w5) + b5)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=hypothesis))
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis = 1))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 4. 모델 훈련

batch_size = 100
total_batch = int(len(x_train/batch_size)) # 60000/100 = 600
epochs = 1000

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(epochs):
    avg_loss=0
    
    for i in range(int(total_batch)): #100개씩 600번 돌기/
        start = i * batch_size        # 0, 10, 200, ...   59900
        end = start + batch_size      # 100,200, 300, ... 60000
    
    _, loss_val = sess.run([train, loss], 
                            feed_dict={x: x_train[start:end], y: y_train[start:end]})

    avg_loss += loss_val/ total_batch
    # -> one epochs
    print("Epochs:", epoch + 1, "Loss: {:.9f}".format(avg_loss))    

print("훈련끝")

# 훈련된 모델을 통해 예측값 출력
y_pred = sess.run(hypothesis, feed_dict={x: x_test})
# print("Predictions:", y_pred)

# 평가 지표 계산
y_pred_label = np.argmax(y_pred, axis=1)
y_test_label = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_label, y_pred_label)
print("Accuracy:", acc)

sess.close()
# # Accuracy: 0.8625