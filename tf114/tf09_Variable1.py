import tensorflow as tf
tf.compat.v1.set_random_seed(123) #compat은 tensorflow114 에서 권고사항 

변수 = tf.compat.v1.Variable(tf.random_normal([1]),name = 'weight') #숫자 한개가 들어있다 2로 바꾸면 숫자 2개가 들어있다 얘는 쉐이프
print(변수)

#초기화 첫번째 방법
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) #초기화
aaa = sess.run(변수)
print('aaa:', aaa)
sess.close()

#초기화 두번째 방법
#변수는 variable에서만 사용한다

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess) #텐서플로 데이터형인 '변수'를 파이썬에서 볼수 있는 놈으로 바꿔준다
print('bbb',bbb)
sess.close()

#초기화 세번째 방법

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print('ccc:',ccc)
sess.close()
