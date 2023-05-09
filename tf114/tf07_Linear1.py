import tensorflow as tf
tf.set_random_seed(337)

#1. 데이터
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111,dtype=tf.float32)
b = tf.Variable(100,dtype=tf.float32) #통상 0임

#2.모델 구성

# y = wx + b
#충격적인 반전 : 원래는 y = xw + b임 그래서 array(행렬)이면 차이가 있음.
hypothesis = x * w + b #hypothesis = loss라고 보면됨. #우리가 예측한 값

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #실직적으로 까보면 웨이트 갱신하는거다
train = optimizer.minimize(loss) 
#경사하강법 방식으로 옵티마이저를 최적화 시켜준다. 한마디로 로스의 최소값을 뽑는다.
# 저거 세줄이 model.compile(loss = 'mse', optimizer = 'sgd') 이거임. SGD는 확률적 경사 하강법(Stochastic Gradient Descent)
# weight갱신되는 식 w - learning_rate*(AE/AW) #learning_rate*(AE/AW) 로스미분한거
#  미분한다는건 그시점의 기울기를 찾겠다

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())# sess을 하면 변수를 초기화 먼저 해줌

#아래가 model.fit이 된다
epochs = 2001

for step in range(epochs):
    sess.run(train)                                                                                                                                                                        
    if step %20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b)) #다 초기화 해야되기때문에
          # verbose 같은거.
sess.close() #수동옵션 저장되는것을 방지 #세션열었으면 닫아줘야한다 