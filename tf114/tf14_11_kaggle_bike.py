import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

#1 데이터
path='./_data/kaggle_bike/'
path_save='./_save/kaggle_bike/'    

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                 

train_csv = train_csv.dropna()   

x = train_csv.drop(['count'], axis=1)   
y = train_csv['count']
print(x.shape, y.shape)#(10886, 10) (10886,)

y = y.values.reshape(-1,1)


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
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5) #adam이 더 좋다

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

# Mean squared error = 147508.3948
# R2 score = -3.4958