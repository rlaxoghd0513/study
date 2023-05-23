# earlystopping 구현
#1. 최소값을 넣은 변수 하나, 카운트할 변수 하나 준비
#2. 다음 에포에 값과 최소값을 비교
#3. 최소값이 갱신되면 그 최솟값을 넣어주고 카운트변수 초기화
#4. 갱신이 안되면 카운트 변수 ++1
#5. 카운트 변수가 내가 원하는 얼리스타핑 갯수에 도달하면 for문을 stop
 
x = 10
y = 10
w = 1111  # Initial value is important, so we perform initialization later
lr = 0.1
epoch = 5000000

min_loss = float('inf')  # Initialize the minimum loss to infinity
count = 0  # Initialize the count variable

for i in range(epoch):
    hypothesis = x * w
    loss = (hypothesis - y) ** 2  # MSE
    print('epoch:', i, 'loss:', round(loss, 4), '\tPredict:', round(hypothesis, 4))
    
    if loss < min_loss:
        min_loss = loss
        count = 0  # Reset the count variable if a new minimum loss is found
    else:
        count += 1  # Increment the count variable if the loss doesn't improve
    
    if count == 30:  # earlystopping patience 30
        break
    
    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2

    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2

    if up_loss >= down_loss:           
        w = w - lr
    else:
        w = w + lr