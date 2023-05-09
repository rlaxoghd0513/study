import tensorflow as tf
tf.set_random_seed(369)

#1. 데이터
x = [1,2,3,4,5]
y = [2,4,6,8,10]

w = tf.Variable(333,dtype=tf.float32)
b = tf.Variable(111,dtype=tf.float32) #통상 0임


####################[실습]###################

#2.모델 구성

# y = wx + b
hypothesis = x * w + b #hypothesis = loss라고 보면됨.

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0169)
train = optimizer.minimize(loss) 

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())# sess을 하면 초기화 먼저 해줌

epochs = 5501

for step in range(epochs):
    sess.run(train)                                                                                                                                                                        
    if step %20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b)) #다 초기화 해야되기때문에
          # verbose 같은거.
sess.close() #수동옵션 저장되는것을 방지
