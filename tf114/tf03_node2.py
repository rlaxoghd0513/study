import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

#실습
#덧셈 node3
#뺄셈 node4
#곱셈 node5
#나누기 node6

sess = tf.compat.v1.Session()

node3 = node1 + node2 
#node = tf.add(node1,node2)
node4 = node1 - node2
#
node5 = node1 * node2
#
node6 = node1 / node2
#

print(sess.run(node3)) # 5.0
print(sess.run(node4)) # -1.0
print(sess.run(node5)) # 6.0
print(sess.run(node6)) # 0.6666667