import tensorflow as tf

x_train = [1]
y_train = [2]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype = tf.float32, name = 'weight') #초기값은 10

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse


##################### 옵티마이저 ###########################

lr = 0.1 #learning_rate

gradient = tf.reduce_mean((x * w - y) * x)   
# gradient = tf.reduce_mean((hypothesis - y) * x)    

descent = w - lr * gradient # w - lr * 미분E/미분W

update = w.assign(descent) # w = w- lr * gradient

###########################################################

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    
    _,loss_v,w_v = sess.run([update, loss, w], feed_dict = {x: x_train, y: y_train}) #update는 안보고 loss변화량과 w변화량만 보겠다
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v) #리스트 값이니까 sess.close해도 안없어진다

sess.close()

print('=============================w history ==================')
print(w_history)
print('====================loss_history================')
print(loss_history)