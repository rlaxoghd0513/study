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
# 시그모이드 
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w)+b) 

#3 컴파일 훈련

# loss = tf.reduce_mean(tf.square(hypothesis - y)) #이진 분류라서 loss='mse'사용불가 
loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis)) #loss='binary_crossentropy'
# y원값이 0이면 앞에껀 통으로 사라지고 뒤에꺼만 돌고 y원값이 1이면 뒤에껀 통으로 사라지고 앞에꺼만 돈다

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)
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
y_predict = tf.sigmoid(tf.matmul(x_test,w_val)+b_val)
y_predict = tf.cast(y_predict>0.5, dtype=tf.float32) #자료형을 바꿔준다 #y_predict가 0.5이상이면 true->float32로 바꾸어라 

# sess.run(y_predict, feed_dict={x_test: x_data})
# 현재 y_predict는 텐서형태고 y_data는 numpy형태라서 둘이 비교가 안된다 #반환값을 설정해줘라

y_pred = sess.run(y_predict, feed_dict={x_test: x_data})
print(type(y_pred))
print('예측값:', y_pred)

acc = accuracy_score(y_data, y_pred)
mse = mean_squared_error(y_data, y_pred)
print('acc:', acc)
print('mse', mse)

# Close the TensorFlow session
sess.close()

# acc: 0.5
# mse 0.5