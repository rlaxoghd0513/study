import tensorflow as tf
tf.compat.v1.set_random_seed(337)

#1 데이터
x_data= [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]] #(6,2)
y_data = [[0],[0],[0],[1],[1],[1]]            #(6,1)

##########################################################
#실습 시그모이드 빼고 걍 만들어봐
#########################################################

x = tf.compat.v1.placeholder(tf.float32, shape =[None,2]) #행의 갯수는 변경될수도 있다 그래서 None으로 명시 
y = tf.compat.v1.placeholder(tf.float32, shape = [None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]), name = 'weight', dtype = tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name = 'bias', dtype = tf.float32)

#2  모델

hypothesis = tf.compat.v1.matmul(x,w)+b 

#3 컴파일 훈련

loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=3e-5)
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=3e-5)
train = optimizer.minimize(loss)

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

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None,2])

# y_predict = x_test*w_val+b_val # 넘파이랑 텐서랑 행렬곱해서 에러 그래서 matmul쓴다
y_predict = tf.matmul(x_test,w_val)+b_val

# sess.run(y_predict, feed_dict={x_test: x_data})
# 현재 y_predict는 텐서형태고 y_data는 numpy형태라서 둘이 비교가 안된다 #반환값을 설정해줘라

y_pred = sess.run(y_predict, feed_dict={x_test: x_data})
print(type(y_pred))
print('예측값:', y_pred)

r2 = r2_score(y_data, y_pred)
mse = mean_squared_error(y_data, y_pred)
print('r2:', r2)
print('mse', mse)

# Close the TensorFlow session
sess.close()

# 예측값: [[ 0.17211509]
#  [ 0.87306345]
#  [-0.16417801]
#  [ 1.116167  ]
#  [ 1.2377187 ]
#  [ 0.77987397]]
# r2: 0.37514785724760225
# mse 0.15621303568809944