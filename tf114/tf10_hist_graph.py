#실습 
#learning_rate 수정해서 epoch 100번 이하로 줄이기
#step = 100 이하 w=1.99, b = 0.99

import tensorflow as tf
tf.compat.v1.set_random_seed(337)
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) #초기화

#2.모델 구성

hypothesis = x * w + b #hypothesis = loss라고 보면됨.

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse 
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.08)
train = optimizer.minimize(loss) 

#3-2. 훈련
loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess:

    sess.run(tf.compat.v1.global_variables_initializer())# sess을 하면 초기화 먼저 해줌

    epochs = 100
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
                                           feed_dict={x:[1,2,3,4,5], y:[2,4,6,8,10]})                                                                                                                                                                 
        if step %20 == 0:
            print(step, loss_val, w_val, b_val) #다 초기화 해야되기때문에
        
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
        
    x_data = [6,7,8]
    x_test  = tf.compat.v1.placeholder(tf.float32, shape=[None]) 
    y_predict = x_test * w_val + b_val
    print('[6,7,8]의 예측 :',sess.run(hypothesis, 
                                           feed_dict={x:x_data}))

print(loss_val_list)
print(w_val_list)

# import matplotlib.pyplot as plt
# plt.plot(loss_val_list) #x나 y만 넣어도 된다
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()

# plt.plot(w_val_list)
# plt.xlabel('epochs')
# plt.ylabel('weights')
# plt.show()

# plt.scatter(w_val_list, loss_val_list, ) #경사하강그림
# plt.xlabel('weights')
# plt.ylabel('loss')
# plt.show()

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 첫 번째 서브플롯: loss_val_list 그래프
axs[0].plot(loss_val_list)
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('loss')

# 두 번째 서브플롯: w_val_list 그래프
axs[1].plot(w_val_list)
axs[1].set_xlabel('epochs')
axs[1].set_ylabel('weights')

# 세 번째 서브플롯: w_val_list vs. loss_val_list 그래프
axs[2].scatter(w_val_list, loss_val_list)
axs[2].plot(w_val_list, loss_val_list)
axs[2].set_xlabel('weight')
axs[2].set_ylabel('loss')

plt.tight_layout()  # 서브플롯 간의 간격 조절
plt.show()

