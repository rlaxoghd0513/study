import tensorflow as tf
import matplotlib.pyplot as plt

x = [1,2]
y = [1,2]
w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis))

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30,51):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w: curr_w})
        
        w_history.append(curr_w)
        loss_history.append(curr_loss)
        
print('======================w_history===================')
print(w_history)
print('=====================loss_history==================')
print(loss_history)

# 2250, 250, 0, 250, 2250, 6250
plt.plot(w_history, loss_history)
plt.xlabel('weights')
plt.ylabel('loss')
plt.show()
# 로스가 가장 낮은 곳으로 간다
# wieght 진행방향이 변화량이 음수가 되거나 양수가 될때마다 변한다



