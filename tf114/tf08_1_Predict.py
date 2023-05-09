import tensorflow as tf
tf.set_random_seed(337)

#1. 데이터

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])
w = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
x_data = tf.placeholder(tf.float32, shape=[None])
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) #초기화

#2.모델 구성

hypothesis = x * w + b #hypothesis = loss라고 보면됨.

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss) 

#3-2. 훈련
with tf.compat.v1.Session() as sess:

    sess.run(tf.global_variables_initializer())# sess을 하면 초기화 먼저 해줌

    epochs = 101
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
                                           feed_dict={x:[1,2,3,4,5], y:[2,4,6,8,10]})                                                                                                                                                                 
        if step %20 == 0:
            print(step, loss_val, w_val, b_val) #다 초기화 해야되기때문에
    x_data = [6,7,8]
    x_test = tf.compat.v1.placeholder(tf.float32,shape=[None])
    y_pred = x_test * w_val +b_val

print('[6,7,8] 예측 :', 
      sess.run(hypothesis, feed_dict={x : x_data}))
    
                
####################실습##########################
# x_data = [6,7,8]
# #예측값을 뽑아라
##################################################