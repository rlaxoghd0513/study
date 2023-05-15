import tensorflow as tf
tf.compat.v1.set_random_seed(337)

x_data = [[73,51,65],
          [92,98,11],
          [89,31,33],
          [99,33,100],
          [17,66,79]]
y_data = [[152],[185],[180],[205],[142]]

x = tf.compat.v1.placeholder(tf.float32, shape =[None,3]) #행의 갯수는 변경될수도 있다 그래서 None으로 명시 
y = tf.compat.v1.placeholder(tf.float32, shape = [None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]), name = 'weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name = 'bias')

#2  모델
# hypothesis = x*w +b
hypothesis = tf.compat.v1.matmul(x,w)+b #행렬곱이 된다  #위에거나 아래 둘중에 선택해서 써라 # 왠만하면 matmul써라

# x.shape = (5,3) y.shape = (5,1)
# hypothesis = x*w+b 
        #    = (5,3)*w+b = (5,1)
#행렬연산 w가 어떻게 되야 (5,1)이 나오나
# (5,3)*(3,1)이 되어야 (5,1)로 나온다

#3 컴파일 훈련
################################### 실습 #########################################
#컴파일하고 r2랑 mse 뽑자

loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=3e-5)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

epochs=2001
for step in range(epochs):
    
    # _,loss_v,w_v = sess.run([update, loss, w], feed_dict = {x: x_train, y: y_train}) #update는 안보고 loss변화량과 w변화량만 보겠다
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                 feed_dict = {x:x_data, y: y_data}) #update와 loss변화량과 w변화량을 보겠다
    if step % 20 ==0:
        print(step, loss_val, w_val, b_val)

y_pred = sess.run(hypothesis, feed_dict={x: x_data})
print('예측값:', y_pred)

# Calculate mean squared error and R-squared score
mse = mean_squared_error(y_data, y_pred)
r2 = r2_score(y_data, y_pred)

# Print the results
print(f"Mean squared error = {mse:.4f}")
print(f"R2 score = {r2:.4f}")

# Close the TensorFlow session
sess.close()



