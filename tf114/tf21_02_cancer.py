import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(337)
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
# One-hot 인코딩

print(x.shape,y.shape) #(569, 30) (569, 2)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([30, 50], dtype=tf.float32), name='weight')
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
layer4 = tf.nn.sigmoid(tf.matmul(layer3, w4) + b4)

w5 = tf.compat.v1.Variable(tf.random.normal([40, 1], dtype=tf.float32), name='weight')
b5 = tf.compat.v1.Variable(tf.zeros([1], dtype=tf.float32), name='bias')
hypothesis = tf.nn.sigmoid(tf.matmul(layer4, w5) + b5)

# 3. 컴파일, 훈련
loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# 4. 모델 훈련
epochs = 10000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs):
        _, loss_val = sess.run([train, loss], feed_dict={x: x_train, y: y_train.reshape(-1, 1)})  # 수정된 부분

        if step % 100 == 0:
            print("Step:", step, "Loss:", loss_val)

    # 훈련된 모델을 통해 예측값 출력
    y_pred = sess.run(hypothesis, feed_dict={x: x_test})
    print("Predictions:", y_pred)

    # 예측값을 0과 1로 이진화
    y_pred_label = np.round(y_pred).astype(int)
    y_test_label = y_test.astype(int)

    accuracy = accuracy_score(y_test_label, y_pred_label)
    print("Accuracy:", accuracy)