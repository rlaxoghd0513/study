import tensorflow as tf

node1 = tf.constant(3.0, tf.float32) #실수형32 의 3.0
node2 = tf.constant(4.0) # float 형태

# node3 = node1 + node2
node3 = tf.add(node1, node2) #위에 것과 동일.

print(node1) #Tensor("Const:0", shape=(), dtype=float32)
print(node2) #Tensor("Const_1:0", shape=(), dtype=float32)
print(node3) #Tensor("add:0", shape=(), dtype=float32)



sess = tf.compat.v1.Session()

print(node3) # Tensor("add:0", shape=(), dtype=float32) 그래프의 모양이 잘나옴.
print(sess.run(node3)) # 7.0
print(sess.run(node1))