#pip install keras==1.2.2
# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
# import keras
# print(keras.__version__) #Using TensorFlow backend. 텐서플로를 바닥에 깔고 시작. 그래서 느림.

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

#[실습] 맹그러

# One-hot 인코딩
encoder = OneHotEncoder(sparse=False)
x_train = x_train.reshape(60000,-1)
x_test = x_test.reshape(10000,-1)
y_train = encoder.fit_transform(y_train.reshape(60000,-1))
y_test = encoder.transform(y_test.reshape(10000,-1))

# print(x_train.shape, y_train.shape) #(60000, 784) (60000, 10)
# print(x_test.shape, y_test.shape)   #(10000, 784) (10000, 10)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

w1 = tf.compat.v1.Variable(tf.random.normal([784, 50], dtype=tf.float32), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([50], dtype=tf.float32), name='bias1')
layer1 = tf.compat.v1.matmul(x, w1) + b1

w2 = tf.compat.v1.Variable(tf.random.normal([50, 40], dtype=tf.float32), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([40], dtype=tf.float32), name='bias2')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.random.normal([40, 40], dtype=tf.float32), name='weight3')
b3 = tf.compat.v1.Variable(tf.zeros([40], dtype=tf.float32), name='bias3')
layer3 = tf.compat.v1.matmul(layer2, w3) + b3

w4 = tf.compat.v1.Variable(tf.random.normal([40, 40], dtype=tf.float32), name='weight4')
b4 = tf.compat.v1.Variable(tf.zeros([40], dtype=tf.float32), name='bias4')
layer4 = tf.nn.softmax(tf.matmul(layer3, w4) + b4)

w5 = tf.compat.v1.Variable(tf.random.normal([40, 10], dtype=tf.float32), name='weight5')
b5 = tf.compat.v1.Variable(tf.zeros([10], dtype=tf.float32), name='bias5')
hypothesis = tf.nn.softmax(tf.matmul(layer4, w5) + b5)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=hypothesis))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 4. 모델 훈련
epochs = 560
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(epochs):
        _, loss_val = sess.run([train, loss], feed_dict={x: x_train, y: y_train.tolist()})

        if epoch % 20 == 0:
            print("Epochs:", epoch, "Loss:", loss_val)

    # 훈련된 모델을 통해 예측값 출력
    y_pred = sess.run(hypothesis, feed_dict={x: x_test})
    # print("Predictions:", y_pred)

    # 평가 지표 계산
    y_pred_label = np.argmax(y_pred, axis=1)
    y_test_label = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_test_label, y_pred_label)
    print("Accuracy:", acc)

# Accuracy: 0.8625