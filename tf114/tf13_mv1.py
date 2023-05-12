import tensorflow as tf
from sklearn.metrics import r2_score

tf.compat.v1.set_random_seed(123)

# Define the input and output data
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 143.]

# Define the model variables
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))

# Define the input placeholders
x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

# Define the model
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# Define the loss and optimizer
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

# Train the model
loss_val_list = []
w_val_list = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    loss_val, hy_val,_ = sess.run([loss, hypothesis, train], feed_dict = {x1:x1_data,x2:x2_data,x3:x3_data,y:y_data})
    if step %100 ==0:
        print(step, 'loss:', loss_val, '\nPreidiction:\n', hy_val)
        
mean_y = tf.reduce_mean(y)
ss_tot = tf.reduce_sum(tf.square(y-mean_y))
ss_res = tf.reduce_sum(tf.square(y - hypothesis))
r_squared = 1 - (ss_res/ss_tot)

print('r2:', sess.run(r_squared, feed_dict={x1:x1_data,x2:x2_data,x3:x3_data}))
sess.close()    
   
