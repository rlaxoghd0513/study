import tensorflow as tf
tf.compat.v1.set_random_seed(337)

x_data = [[73,51,65],
          [92,98,11],
          [89,31,33],
          [99,33,100],
          [17,66,79]]
y_data = [[152],[185],[180],[205],[142]]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([]), name = 'weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([]), name = 'bias')

#2  모데ㅐㄹ
hypothesis = x*w +b

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

print(sess.run([hypothesis,w,b], feed_dict = {x:x_data}))

# [array([[25.064732 , 17.198214 , 22.20418  ],
#        [31.858541 , 34.003956 ,  2.8954527],
#        [30.785837 , 10.046833 , 10.7619705],
#        [34.361523 , 10.7619705, 34.719093 ],
#        [ 5.040867 , 22.561749 , 27.210146 ]], dtype=float32), 0.357569, -1.0378063]