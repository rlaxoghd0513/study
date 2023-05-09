import tensorflow as tf
print(tf.__version__)

print("Hello World")

aaa = tf.constant('Hello world') #상수(constant) 바뀌지 않는 숫자.

print(aaa)            #Tensor("Const:0", shape=(), dtype=string)

#https://www.youtube.com/watch?v=-57Ne86Ia8w 10분쯤 나오는 사진.
#sess.run을 넣어야함. 그래프연산방식이기때문에!!

# sess = tf.Session() #이게 추가가 됨.
sess = tf.compat.v1.Session()
print(sess.run(aaa)) #b'Hello world' b는 바이너리.  
#아웃풋 출력시킬때마다 sess.run을 넣어야됨