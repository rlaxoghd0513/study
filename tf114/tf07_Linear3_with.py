import tensorflow as tf
tf.set_random_seed(337)

#1. 데이터
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111,dtype=tf.float32)
b = tf.Variable(0,dtype=tf.float32) #통상 0임

#2.모델 구성

# y = wx + b
#충격적인 반전 : 원래는 y = xw + b임 그래서 array(행렬)이면 차이가 있음.
hypothesis = x * w + b #hypothesis = loss라고 보면됨.

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss) 
#경사하강법 방식으로 옵티마이저를 최적화 시켜준다. 한마디로 로스의 최소값을 뽑는다.
# 저거 세줄이 model.compile(loss = 'mse', optimizer = 'sgd') 이거임. SGD는 확률적 경사 하강법(Stochastic Gradient Descent)

#3-2. 훈련
with tf.compat.v1.Session() as sess:

# sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())# sess을 하면 초기화 먼저 해줌

    epochs = 2001

    for step in range(epochs):
        sess.run(train)                                                                                                                                                                        
        if step %20 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b)) #다 초기화 해야되기때문에
            # verbose 같은거.
    # sess.close() #수동옵션 저장되는것을 방지