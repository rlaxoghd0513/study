import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
 #GridSearch 파라미터의 경우의 수만큼 가둔다

x,y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state = 333, test_size = 0.2)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle= True, random_state=333)


parameters = [                                                     #리스트 안에 딕셔너리 형태
    {"C":[1,10,100,1000], "kernel":['linear'],'degree':[3,4,5]},   #12번 돈다    약간씩 자기가 하고 싶은 대로 수정 가능한것  degree디폴트3
    {"C":[1,10,100],'kernel':['rbf','linear'],'gamma':[0.001,0.0001]},      #12번 돈다
    {"C":[1,10,100,1000], 'kernel':['sigmoid'],                    #24번 돈다
     'gamma':[0.01,0.001,0.0001], 'degree':[3,4]},
    {"C":[0.1,1], "gamma":[1,10]}                                  #4번 돈다
    ]                                                              #총 52번 돈다

#모델구성
model = GridSearchCV(SVC(), parameters,
                    #  cv=kfold,     #cv=5도 가능   kfold에 stratify가 안들어가있는데 5로 하니까 더 좋아졌다 디폴트로 stratify가 들어가 있단 얘기
                     cv=5,            #분류의 디폴트는 StratifiedKFold이다 
                     verbose=1,
                     refit=True,     #디폴트 True
                    #  refit=False,    # refit은 최적의 값을 계속 저장하고 있다 false 로 하면 260번을 돌긴 하지만 마지막값을 저장하기 때문에 최적의 값을 저장하지 않는다 
                     n_jobs=-1) #gridsearch1에서 for문으로 길게 돌린게 한줄로 정리된다  kfold했기때문에 5번 더 돌아서 240번 돈다

#컴파일 훈련
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수:", model.best_estimator_)   #전체 파라미터가 나오고
#최적의 매개변수: SVC(C=1000, gamma=0.001, kernel='sigmoid')
print("최적의 파라미터:", model.best_params_)      #내가 지정한 것만 나온다
#최적의 파라미터: {'C': 1000, 'degree': 3, 'gamma': 0.001, 'kernel': 'sigmoid'}
print("best_score:", model.best_score_)
#best_score: 0.9833333333333334
print("model.score:", model.score(x_test, y_test))
#model.score: 0.9666666666666667

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc:", acc)
# acc: 0.9666666666666667
y_pred_best = model.best_estimator_.predict(x_test) #모델의 최적의 평가값에 프리딕트. 우리가 만든 model이란 클래스에 best_estimator는 프리딕트를 가지고 있다
print("최적 튠:",accuracy_score(y_test,y_pred_best))  #model.predict를 쓰든 model.best_estimator_.predict 를 쓰든 맘에 드는거 써라
# 최적 튠: 0.9666666666666667
print("걸린시간:", round(end_time-start_time,2), '초')
# 걸린시간: 2.4 초

# gamma: RBF (Radial basis function) 커널에서 사용되는 하이퍼파라미터로, 커널 함수의 영향 범위를 조절합니다.
# gamma 값이 작을수록 커널 함수의 영향 범위가 크며, 클수록 작아집니다.

# degree: 다항 커널(Polynomial kernel)에서 사용되는 하이퍼파라미터로, 다항 커널의 차수를 지정합니다.
# 일반적으로 1 이상의 정수 값을 사용하며, 높은 차수일수록 더 복잡한 결정 경계를 생성합니다. 기본값은 3입니다.

# C: 소프트 마진(Soft margin) SVM에서 사용되는 하이퍼파라미터로, 각 데이터 포인트의 분류를 얼마나 엄격하게 할 것인지를 조절합니다. 
# C 값이 작을수록 분류가 더 허용되며, 클수록 분류가 더 엄격해집니다.

# kernel: SVM에서 사용할 커널 함수를 선택하는 하이퍼파라미터로, 데이터를 고차원 공간으로 매핑하여 비선형 결정 경계를 생성합니다. 
# 주요 커널 함수로는 선형 커널(Linear kernel), 다항 커널(Polynomial kernel), RBF 커널(Radial basis function kernel) 등이 있습니다.
        
