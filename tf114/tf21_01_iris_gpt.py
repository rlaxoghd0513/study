import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터
x, y = load_iris(return_X_y=True)
# One-hot 인코딩
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w1 = tf.compat.v1.Variable(tf.random.normal([4, 50], dtype=tf.float32), name='weight')
b1 = tf.compat.v1.Variable(tf.zeros([50], dtype=tf.float32), name='bias')
layer1 = tf.compat.v1.matmul(x, w1) + b1

w2 = tf.compat.v1.Variable(tf.random.normal([50, 40], dtype=tf.float32), name='weight')
b2 = tf.compat.v1.Variable(tf.zeros([40], dtype=tf.float32), name='bias')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.random.normal([40, 40], dtype=tf.float32), name='weight')
b3 = tf.compat.v1.Variable(tf.zeros([40], dtype=tf.float32), name='bias')
layer3 = tf.compat.v1.matmul(layer2, w3) + b3

w4 = tf.compat.v1.Variable(tf.random.normal([40, 40], dtype=tf.float32), name='weight')
b4 = tf.compat.v1.Variable(tf.zeros([40], dtype=tf.float32), name='bias')
layer4 = tf.nn.softmax(tf.matmul(layer3, w4) + b4)

w5 = tf.compat.v1.Variable(tf.random.normal([40, 3], dtype=tf.float32), name='weight')
b5 = tf.compat.v1.Variable(tf.zeros([3], dtype=tf.float32), name='bias')
hypothesis = tf.nn.softmax(tf.matmul(layer4, w5) + b5)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=hypothesis))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.009)
train = optimizer.minimize(loss)

# 4. 모델 훈련
epochs = 18000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs):
        _, loss_val = sess.run([train, loss], feed_dict={x: x_train, y: y_train})

        if step % 100 == 0:
            print("Step:", step, "Loss:", loss_val)

    # # 훈련된 모델을 통해 예측값 출력
    # y_pred_train = sess.run(hypothesis, feed_dict={x: x_train})
    # y_pred_train = np.argmax(y_pred_train, axis=1)

    y_pred_test = sess.run(hypothesis, feed_dict={x: x_test})
    y_pred_test = np.argmax(y_pred_test, axis=1)

    # 정확도 계산
    # acc_train = accuracy_score(np.argmax(y_train, axis=1), y_pred_train)
    acc_test = accuracy_score(np.argmax(y_test, axis=1), y_pred_test)

    # print("Train Accuracy:", acc_train)
    print("Test Accuracy:", acc_test)