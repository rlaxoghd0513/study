import tensorflow as tf
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

tf.compat.v1.set_random_seed(123)

# Define the input and output data
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

# Define the model variables
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
# w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype = tf.float32)
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype = tf.float32)
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype = tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype = tf.float32)

# Define the input placeholders
x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

# Define the model
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

# Define the loss and optimizer
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# Train the model

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=2001
for step in range(epochs):
    
    # _,loss_v,w_v = sess.run([update, loss, w], feed_dict = {x: x_train, y: y_train}) #update는 안보고 loss변화량과 w변화량만 보겠다
    cost_val,_ = sess.run( [loss,train],
                                 feed_dict = {x1:x1_data,x2:x2_data, x3:x3_data, y: y_data}) #update와 loss변화량과 w변화량을 보겠다
    if step % 20 ==0:
        print(epochs, 'loss:', cost_val)

sess.close()

   
