# pip install keras==1.2.2
# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from keras.utils.np_utils import to_categorical
import keras
import tensorflow as tf
import numpy as np

# print(keras.__version__)

# 1 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 784)/255. # 데이터의 구조만 바뀐다. 순서와 값은 바뀌지 않는다.
# x_train = x_train.reshape(60000, 784).astype('float32')/255. # 위에거랑 같음
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])/255.

y_train = to_categorical(y_train)   
y_test = to_categorical(y_test)

# 2 모델 구성
x_p = tf.compat.v1.placeholder(tf.float32, shape = [None, 784])
# x_p = tf.compat.v1.placeholder('float', shape = [None, 784]) # 이거도 된다
y_p = tf.compat.v1.placeholder(tf.float32, shape = [None, 10])

w1 = tf.compat.v1.Variable(tf.random_normal([784, 20]), name = 'weight1') # 정규 분포
# w1 = tf.compat.v1.Variable(tf.random_uniform([784, 100]), name = 'weight1') # 균등 분포
b1 = tf.compat.v1.Variable(tf.zeros([20]), name = 'bias1')
layer1 = tf.nn.relu(tf.compat.v1.matmul(x_p, w1) + b1)
dropout1 = tf.compat.v1.nn.dropout(layer1, rate = 0.2)

w2 = tf.compat.v1.Variable(tf.random_normal([20, 20]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.zeros([20]), name = 'bias2')
layer2 = tf.nn.selu(tf.compat.v1.matmul(dropout1, w2) + b2)
dropout2 = tf.compat.v1.nn.dropout(layer2, rate = 0.2)

w3 = tf.compat.v1.Variable(tf.random_normal([20, 20]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.zeros([20]), name = 'bias3')
layer3 = tf.compat.v1.matmul(dropout2, w3) + b3

w4 = tf.compat.v1.Variable(tf.random_normal([20, 20]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.zeros([20]), name = 'bias4')
layer4 = tf.compat.v1.matmul(layer3, w4) + b4

w5 = tf.compat.v1.Variable(tf.random_normal([20, 20]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.zeros([20]), name = 'bias5')
layer5 = tf.compat.v1.matmul(layer4, w5) + b5

w6 = tf.compat.v1.Variable(tf.random_normal([20, 20]), name = 'weight6')
b6 = tf.compat.v1.Variable(tf.zeros([20]), name = 'bias6')
layer6 = tf.compat.v1.matmul(layer5, w6) + b6

w7 = tf.compat.v1.Variable(tf.random_normal([20, 10]), name = 'weight7')
b7 = tf.compat.v1.Variable(tf.zeros([10]), name = 'bias7')
# layer7 = tf.compat.v1.matmul(layer6, w7) + b7
hypothesis = tf.nn.softmax(tf.matmul(layer6, w7) + b7)

# 3 컴파일
# logits = layer7
# hypothesis = tf.nn.softmax(tf.matmul(layer6, w7) + b7)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(hypothesis, y_p))
# loss = -tf.reduce_mean(y_p * tf.log(hypothesis + 1e-5) + (1 - y_p) * tf.log(1 - hypothesis + 1e-5))

# train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.0001).minimize(loss)
train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate = 1e-5).minimize(loss)

sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 100

epochs = 1001

total_batch = int(len(x_train/batch_size)) # 60000/100 = 600

# 4 평가
for s in tqdm(range(epochs)):
    avg_loss = 0
    for b in tqdm(range(int(total_batch))): # 100개씩 600번 돈다
        start = b * batch_size # 0, 100, 200, .....
        end = start + batch_size # 100, 200, 300, .....
        # x_train[:600], y_train[:600]
        _, loss_val, w_val, b_val = sess.run([train, loss, w7, b7], feed_dict = {x_p:x_train[start:end], y_p:y_train[start:end]})
        
        avg_loss += loss_val / total_batch
    print('Epoch : ', s + 1, 'loss : , {:.9f}'.format(avg_loss))
print('훈련 끝')
# if s % 20 == 0:
print(f'{s}번째, loss : {loss_val}')
    
y_predict = sess.run(hypothesis, feed_dict = {x_p:x_test})

y_predict_class = np.argmax(y_predict, axis=1)

y_test_class = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_class, y_predict_class)
print('acc : ', acc)