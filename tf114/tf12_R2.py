import tensorflow as tf

x_train = [1,2,3]
y_train = [1,2,3]
x_test = [4,5,6]
y_test = [4,5,6]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype = tf.float32, name = 'weight') #초기값은 10

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse


##################### 옵티마이저 ###########################

lr = 0.1 #learning_rate

gradient = tf.reduce_mean((x * w - y) * x)   # loss변화량 한마디로 loss의 미분값이다, 웨이트갱신을 위하여.  앞에 2곱하는건 어차피 loss에 다 똑같이 곱해지니까 생략했다
# 편미분 내가 미분할 놈 제외하곤 상수로 본다
# 체인룰 = 통으로 미분한 값에 그 통 안에 있는 값도 미분해서 곱한다 g = (2x + y)2제곱 , 2x+y = h -> g = h2제곱 -> 그냥 전개해서 풀면 8x+4y, 체인룰해서 풀면 2(2x+y)*2 
# gradient = tf.reduce_mean((hypothesis - y) * x)    

descent = w - lr * gradient # w - lr * 미분E/미분W

update = w.assign(descent) # w = w- lr * gradient w에 descent값을 계속 갱신하지만 연산그래프에 추가만 됐을 뿐이기 때문에 실행시키려면 sess.run(update)를 해주려고 update라는 변수명을 붙였다

###########################################################
up_history = []
w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    
    # _,loss_v,w_v = sess.run([update, loss, w], feed_dict = {x: x_train, y: y_train}) #update는 안보고 loss변화량과 w변화량만 보겠다
    up,loss_v,w_v = sess.run([update, loss, w], feed_dict = {x: x_train, y: y_train}) #update와 loss변화량과 w변화량을 보겠다
    print(step, '\t',up,'\t', loss_v, '\t', w_v) # 왜 up하고 w_v하고 똑같은가요?
    
    up_history.append(up)
    w_history.append(w_v)
    loss_history.append(loss_v) #리스트 값이니까 sess.close해도 안없어진다

sess.close()

print('========================up_history==============')
print(up_history)
print('=============================w history ==================')
print(w_history)
print('====================loss_history================')
print(loss_history)
# up_history랑 w_history랑 똑같다

#########실습 R2, mae 만들어라############
from sklearn.metrics import r2_score, mean_absolute_error

y_predict = x_test * w_v
print(y_predict)#[4.00006676 5.00008345 6.00010014]
r2 = r2_score(y_predict, y_test) #앞뒤 바뀌어도 상관 x
print('r2:', r2)

mae = mean_absolute_error(y_predict, y_test)
print('mae:', mae)