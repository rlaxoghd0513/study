x = 10
y = 10
w = 1111 #초기값도 중요 그래서 나중에 initializer를 한다
lr = 0.1
epoch = 5000000

for i in range(epoch):
    hypothesis = x * w
    loss = (hypothesis - y) ** 2 #mse
    print('loss:', round(loss,4),'\tPredict:', round(hypothesis,4))
    up_predict = x * (w+lr)
    up_loss = (y - up_predict)**2

    down_predict = x * (w - lr)
    down_loss = (y - down_predict)**2

    if(up_loss >= down_loss):
        w = w - lr
    else:
        w = w+lr