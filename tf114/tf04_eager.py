import tensorflow as tf
print(tf.__version__) #1.14.0

# 즉시실행모드!!
print(tf.executing_eagerly()) #False #tensorflow2로 가상환경을 바꾸면 True
#tensorflow1에서는 그래프를 그리기때문에 즉시실행모드가 아님.
sess = tf.compat.v1.Session()
aaa = tf.constant('hello world')

tf.compat.v1.disable_eager_execution() #즉시 실행모드 끄기./ 텐서2.0을 1.0 방식으로
tf.compat.v1.enable_eager_execution()
print(tf.executing_eagerly())
print(tf.executing_eagerly())

print(sess.run(aaa)) 
print(aaa)