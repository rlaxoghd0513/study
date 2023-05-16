import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits

x_data, y_data = load_digits(return_X_y=True)
print(x_data.shape, y_data.shape)#(1797, 64) (1797,)
y_data = y_data.reshape(-1,1)

#2 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None,64])
y = tf.compat.v1.placeholder(tf.float32, shape=(None,1))

w = tf.compat.v1.Variable(tf.random_normal([64,1]), name = 'weight')
b = tf.compat.v1.Variable(tf.zeros([1,1]), name = 'bias')

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x,w) +b)

#3-1 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=3e-5)
# train = optimizer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss) #optimizer랑 train 한줄로

#실습
#3-2
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import numpy as np

epochs=2001
for step in range(epochs):
    
    # _,loss_v,w_v = sess.run([update, loss, w], feed_dict = {x: x_train, y: y_train}) #update는 안보고 loss변화량과 w변화량만 보겠다
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                 feed_dict = {x:x_data, y: y_data}) #update와 loss변화량과 w변화량을 보겠다
    if step % 20 ==0:
        print(step, loss_val, w_val, b_val)

y_predict = sess.run(hypothesis, feed_dict={x:x_data})
y_predict_arg = sess.run(tf.argmax(y_predict, 1))

y_data_arg = np.argmax(y_data,1)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data_arg, y_predict_arg)
print(acc) #1.0