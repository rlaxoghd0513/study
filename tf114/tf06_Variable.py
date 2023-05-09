import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)
y = tf.Variable([3], dtype=tf.float32) #variable는 변수 초기화를 해줘야됨.
#[]한 이유 tensorflow는 행렬연산이기 때문에.

init = tf.compat.v1.global_variables_initializer()
sess.run(init) #sess를 안하면 다 그래프형태로 나오기 때문에 무조건 sess를 넣어야함.

print(sess.run(x + y)) #[5.]

#x,y= 딕셔너리 ,w와 b = variable