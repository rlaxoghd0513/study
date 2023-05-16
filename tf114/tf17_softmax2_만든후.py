import tensorflow as tf
import numpy as np
tf.set_random_seed(337)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

#2 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None,4])
y = tf.compat.v1.placeholder(tf.float32, shape=(None,3))

w = tf.compat.v1.Variable(tf.random_normal([4,3]), name = 'weight')
b = tf.compat.v1.Variable(tf.zeros([1,3]), name = 'bias')

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
print(acc) #0.375