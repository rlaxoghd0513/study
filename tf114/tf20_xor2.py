import tensorflow as tf
import numpy as np
tf.set_random_seed(337)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32) #(4,2)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32) #(4,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])


#2.모델
# model.add(Dense(10, input_shape =2))
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 10]), name='weight1') 
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias1') #dense의 개수랑 bias개수를 동일하게 감.
layer1 = tf.compat.v1.matmul(x, w1) + b1
# layer1 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w1) + b1)

# model.add(Dense(7)
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 7]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([7]), name='bias2')
layer2 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer1, w2)+ b2)

# model.add(Dense(1, activation = 'sigmoid')
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([7, 1]), name='weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias3')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer2, w3)+ b3)

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

predicted = tf.cast(hypothesis > 0.5,dtype=tf.float32) #cast -> True아니면 False
accuracy =tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))
#accuracy = false, True로 반환 -> float32 1.0, 2.0 이렇게됨. -> 숫자의 나누기 4가 들어가서 0.5가됨.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(1501):
        loss_val, _ = sess.run([loss, train], feed_dict = {x: x_data, y : y_data})
        
        if  epoch % 100 == 0:
            print(epoch, loss_val)
            
    h, p, a =sess.run([hypothesis, predicted, accuracy],
             feed_dict = {x : x_data, y : y_data})
    print("예측값 :", h, "\n 원래 값 :", p, "\n Accuracy :", a)