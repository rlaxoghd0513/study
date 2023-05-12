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
# wx와 xw는 완전히 다르다 행렬계산을 생각해보자

hypothesis = x * w + b #hypothesis = loss라고 보면됨. #우리가 예측한 값

#3-1. 컴파일
# **** 오차 = 에러 = cost = loss *****
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse  #reduce_mean 전부 더해서 평균낸다  #square 는 제곱한다

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #실직적으로 까보면 웨이트 갱신하는거다
train = optimizer.minimize(loss) 
#경사하강법 방식으로 옵티마이저를 최적화 시켜준다. 한마디로 로스의 최소값을 뽑는다.
# 저거 세줄이 model.compile(loss = 'mse', optimizer = 'sgd') 이거임. SGD는 확률적 경사 하강법(Stochastic Gradient Descent)
# weight갱신되는 식 w - learning_rate*(AE/AW) #learning_rate*(AE/AW) 로스미분한거
#  미분한다는건 그시점의 기울기를 찾겠다
# 미분 = 그 지점의 변화량 이라고 생각하자

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())# sess을 하면 변수를 초기화 먼저 해줌 # 실질적 초기화 대상 : w ,b

#아래가 model.fit이 된다
epochs = 2001

for step in range(epochs):
    sess.run(train)                                                                                                                                                                        
    if step %20 == 0:   #20번마다 출력한다
        print(step, sess.run(loss), sess.run(w), sess.run(b)) 
          # verbose 같은거.
sess.close() #수동옵션 저장되는것을 방지 #세션열었으면 닫아줘야한다 