import tensorflow as tf
tf.compat.v1.set_random_seed(337)

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#1 데이터
x,y = load_diabetes(return_X_y=True)
# print(x.shape, y.shape) #(442, 10) (442,)
# print(y[:10]) #[151.  75. 141. 206. 135.  97. 138.  63. 110. 310.]

y = y.reshape(-1,1) # (442,1)이 된다 #웨이트와 행렬연산을 해야하기 때문이다
#x(442,10)*w(?,?) +b(?) = y(442,1)    w(?,?)->(10,1) 

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=42)
print(x_train.shape, y_train.shape) #(353, 10) (353, 1)
print(x_test.shape, y_test.shape) #(89, 10) (89, 1)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None,10])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1], dtype = tf.float64), name = 'weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1],dtype=tf.float64), name = 'bias') #bias는 통상 0으로 주니까

#2 모델구성

hypothesis = tf.compat.v1.matmul(x,w) + b

#3-1 컴파일 
loss = tf.reduce_sum(tf.square(hypothesis-y))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001) #웨이트경신되는거 w= w-lr*뭐시기
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001) #adam이 더 좋다

train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=101
for step in range(epochs):
    
    # _,loss_v,w_v = sess.run([update, loss, w], feed_dict = {x: x_train, y: y_train}) #update는 안보고 loss변화량과 w변화량만 보겠다
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                 feed_dict = {xp:x, yp:y}) #update와 loss변화량과 w변화량을 보겠다
    if step % 20 ==0:
        print(step, loss_val, w_val, b_val)

y_pred = sess.run(hypothesis, feed_dict={xp: x})
print('예측값:', y_pred)

# Calculate mean squared error and R-squared score
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print the results
print(f"Mean squared error = {mse:.4f}")
print(f"R2 score = {r2:.4f}")

# Close the TensorFlow session
sess.close()

# Mean squared error = 29072.0555
# R2 score = -3.9026