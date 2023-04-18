import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV #halving 뜻: 이등분, n빵 n빵의 기준은 factor
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
 #GridSearch 파라미터의 경우의 수만큼 가둔다

x,y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state = 1232, test_size = 0.2)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle= True, random_state=3313)


parameters = [                                                     #리스트 안에 딕셔너리 형태
    {"C":[1,10,100,1000], "kernel":['linear'],'degree':[3,4,5]},   #12번 돈다    약간씩 자기가 하고 싶은 대로 수정 가능한것  degree디폴트3
    {"C":[1,10,100],'kernel':['rbf','linear'],'gamma':[0.001,0.0001]},      #12번 돈다
    {"C":[1,10,100,1000], 'kernel':['sigmoid'],                    #24번 돈다
     'gamma':[0.01,0.001,0.0001], 'degree':[3,4]},
    {"C":[0.1,1], "gamma":[1,10]}                                  #4번 돈다
    ]                                                              #총 52번 돈다  cv해서 52*5=260 260번 돈다

#모델구성
#그리드서치에서 랜덤하게 몇개만 빼서 쓰는게 랜더마이즈서치
# model = GridSearchCV(SVC(), parameters,
model = HalvingGridSearchCV(SVC(), parameters,            #cross_val 당 52 중 10개만 랜덤하게 뽑겠다 데이터가 크다 싶으면 randomized 데이터가 작아서 금방 끝나겠다 싶으면 gridsearch
                    #  cv=kfold,     #cv=5도 가능   kfold에 stratify가 안들어가있는데 5로 하니까 더 좋아졌다 디폴트로 stratify가 들어가 있단 얘기
                     cv=5,            #분류의 디폴트는 StratifiedKFold이다 
                     verbose=1,
                     refit=True,     #디폴트 True
                    #  refit=False,    # refit은 최적의 값을 계속 저장하고 있다 false 로 하면 260번을 돌긴 하지만 마지막값을 저장하기 때문에 최적의 값을 저장하지 않는다 
                    #  n_iter=5,     #randomizedSearch에서만 사용 gridsearch는 전부 사용하기때문에 랜덤하게 뺄게 없다
                     factor=3.5, #디폴트3, 소숫점 가능
                     n_jobs=-1) #gridsearch1에서 for문으로 길게 돌린게 한줄로 정리된다  kfold했기때문에 5번 더 돌아서 240번 돈다

#컴파일 훈련
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
print("걸린시간:", round(end_time-start_time,2), '초')

# print(x.shape, x_train.shape) #(1797, 64) (1437, 64)   
#iteration반복 

# (tf274gpu) C:\study> c: && cd c:\study && cmd /C "C:\Users\aiacademy\anaconda3\envs\tf274gpu\python.exe c:\Users\aiacademy\.vscode\extensions\ms-python.python-2023.6.1\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher 54948 -- c:\study\ml\m14_HalvingGridSearch1.py "
# n_iterations: 3                                                  세번째 훈련까지 들어간다
# n_required_iterations: 4
# n_possible_iterations: 3
# min_resources_: 100                                              #최소훈련데이터갯수
# max_resources_: 1437                                             #최대훈련데이터갯수  max_resources_: 1437 훈련시킬수있는 최대 자원 1437개
# aggressive_elimination: False
# factor: 3                                                        factor 디폴트: 3 , n빵할 양
# ----------
# iter: 0
# n_candidates: 52                                                  전체 파라미터 갯수
# n_resources: 100                                                  #0번째 훈련에서 쓸 훈련데이터 갯수
# Fitting 5 folds for each of 52 candidates, totalling 260 fits   
# ----------
# iter: 1
# n_candidates: 18                                              #전체파라미터갯수/factor, 이미 한번 훈련 돌았고 52개의 파라미터 중 factor로 나눈 갯수만큼 상위 파라미터18개를 사용하겠다
# n_resources: 300                                                   #min_resources*factor
# Fitting 5 folds for each of 18 candidates, totalling 90 fits    
# ----------
# iter: 2
# n_candidates: 6                                                   #18/factor   18은 iter:1의 n_candidates  
# n_resources: 900                                                  #300*factor
# Fitting 5 folds for each of 6 candidates, totalling 30 fits      n_resources 합이 max_resourses를 넘지 않게, 너무 모자르지 않게 계산해야한다
# 걸린시간: 3.36 초
# (1797, 64) (1437, 64


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

import pandas as pd      #아무거나 판다스 갖다붙이면 안되고 데이터 모양 보고 해라
# print(pd.DataFrame(model.cv_results_))    #가로 세로 있는거 판다스 데이터 프레임    # 1차원 형태 하나의 행 한가지 한가지 리스트는 벡터 형태 판다스 리스트
# #[52 rows x 17 columns] 52번 돌렸고 그안에 17가지의 칼럼들이 뽑힌다
# print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score'))#값 순서대로 정렬 sort_index는 인덱스 순으로 정렬 (오름차순) 디폴트값
# print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=False))   #(내림차순)
# print(pd.DataFrame(model.cv_results_).columns) #칼럼이 뭐가 있는지 궁금하다 

path = './temp/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True)\
    .to_csv(path+'m14_HalvingGridSearch1.csv')
    
#factor : 3.5
# 걸린시간: 3.83 초
# 최적의 매개변수: SVC(C=100, gamma=0.001)
# 최적의 파라미터: {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
# best_score: 0.9917524799244214
# model.score: 0.9944444444444445
# acc: 0.9944444444444445
# 최적 튠: 0.9944444444444445
# 걸린시간: 3.83 초