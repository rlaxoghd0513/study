import tensorflow as tf
import numpy as np
tf.set_random_seed(337)
from sklearn.metrics import accuracy_score

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32) #(4,2)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32) #(4,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32)

#2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w)+ b)

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))

optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=1e-6)
train = optimizer.minimize(loss)

predicted = tf.cast(hypothesis > 0.5,dtype=tf.float32) #cast -> True아니면 False
accuracy =tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))
#accuracy = false, True로 반환 -> float32 1.0, 2.0 이렇게됨. -> 숫자의 나누기 4가 들어가서 0.5가됨.


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(7501):
        loss_val, _ = sess.run([loss, train], feed_dict = {x: x_data, y : y_data})
        
        if  epoch % 100 == 0:
            print(epoch, loss_val)
            
    h, p, a =sess.run([hypothesis, predicted, accuracy],
             feed_dict = {x : x_data, y : y_data})
    print("예측값 :", h, "\n 원래 값 :", p, "\n Accuracy :", a)

# 예측값 : [[0.44682083]
#         [0.54059273]
#         [0.45738983]
#         [0.55116975]]
#  원래 값 : [[0.]
#             [1.]
#             [0.]
#             [1.]]
#  Accuracy : 0.5

# #3-2. 훈련
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# epochs = 5000

# for epoch in range(epochs):
#     cost_val, _ , w_val, b_val= sess.run([loss, train, w, b], 
#                                 feed_dict={x: x_data, y: y_data})
        
#     if epoch % 20 == 0:
#         print(epoch, 'loss : ', cost_val)

# #4. 평가, 예측            
# x_test = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])

# y_predict = tf.sigmoid(tf.matmul(x_test, w_val)) + b_val

# y_predict = tf.cast(y_predict > 0.5, dtype=tf.float32)

# y_aaa = sess.run(y_predict, feed_dict={x_test:x_data})

# acc = accuracy_score(y_aaa, y_data)
# print('acc : ', acc)

# sess.close()