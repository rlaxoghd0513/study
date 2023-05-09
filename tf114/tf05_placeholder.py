import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly()) #true

#즉시실행모드 해제
tf.compat.v1.disable_eager_execution() #즉시실행모드 해제!
print(tf.executing_eagerly()) #False

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32) #인풋기능만 있다 그래서 feed_dict이 항상 따라다닌다
b = tf.compat.v1.placeholder(tf.float32) #placeholder : 어떤공간에 값을 받는곳(빈방 같은곳)

add_node = a + b
# print(add_node) #Tensor("add_1:0", dtype=float32)
print(sess.run(add_node,feed_dict={a:3, b:4.5})) #7.5
print(sess.run(add_node,feed_dict={a:[1,3], b:[2,4]})) #[3. 7.]

add_and_triple =  add_node * 3
print(add_and_triple) #Tensor("mul:0", dtype=float32)

print(sess.run(add_and_triple, feed_dict={a:7, b:3}))#여기서 실행
print(sess.run(add_and_triple, feed_dict={a:[1,3], b:[2,4]}))