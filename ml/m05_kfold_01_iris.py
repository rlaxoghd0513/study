import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


#1 데이터
x,y = load_iris(return_X_y = True)

# x_train, x_test, y_train, y_test = train_test_split(x,y,shuffle= True, random_state=123, test_size = 0.2)
n_splits = 5
kfold = KFold(n_splits = n_splits , shuffle=True, random_state=123 )  #5개로 나눈다 그러면 데이터는 20프로씩 나눠진다 데이터 모이는걸 방지하려고 shuffle
# kfold = KFold() #이렇게 해도 된다 디폴트값 있다

#2 모델구성
model = LinearSVC()

#3,4 컴파일 훈련 평가 예측

scores = cross_val_score(model, x, y, cv = kfold, n_jobs = 4) #(모델, 데이터, 데이터, 크로스 발리데이션을 어떻게 할건지, cpu코어사용갯수)
# scores = cross_val_score(model, x, y, cv = 5) #이렇게 kfold를 따로 정의하지 않고도 가능하다
print(scores)

#[0.96666667 1.         0.93333333 0.93333333 0.9       ]
#다섯번 훈련을 시켰으니까 5개의 값이 나온다

print('acc:', scores, '\n cross_val_score 평균:', round(np.mean(scores),4)) #\n 줄바꿔서 scores를 넘파이배열로 바꿔서 소숫점 4번째 자리까지 반올림

