import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV #halving 뜻: 이등분, n빵 n빵의 기준은 factor
from sklearn.metrics import accuracy_score,r2_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd

path = './_data/ddarung/'
path_save= './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0) 
train_csv = train_csv.dropna() #dropna 결측치삭제

x = train_csv.drop(['count'], axis=1) 
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state = 1232, test_size = 0.2)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle= True, random_state=3313)


parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,10,12]}
]                     

#모델구성
#그리드서치에서 랜덤하게 몇개만 빼서 쓰는게 랜더마이즈서치
# model = GridSearchCV(SVC(), parameters,
model = HalvingGridSearchCV(RandomForestRegressor(), parameters,            #cross_val 당 52 중 10개만 랜덤하게 뽑겠다 데이터가 크다 싶으면 randomized 데이터가 작아서 금방 끝나겠다 싶으면 gridsearch
                    #  cv=kfold,     #cv=5도 가능   kfold에 stratify가 안들어가있는데 5로 하니까 더 좋아졌다 디폴트로 stratify가 들어가 있단 얘기
                     cv=5,            #분류의 디폴트는 StratifiedKFold이다 
                     verbose=1,
                     refit=True,     #디폴트 True
                    #  refit=False,    # refit은 최적의 값을 계속 저장하고 있다 false 로 하면 260번을 돌긴 하지만 마지막값을 저장하기 때문에 최적의 값을 저장하지 않는다 
                     factor=3.5, #디폴트3, 소숫점 가능
                     n_jobs=-1) #gridsearch1에서 for문으로 길게 돌린게 한줄로 정리된다  kfold했기때문에 5번 더 돌아서 240번 돈다

#컴파일 훈련
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


print("최적의 매개변수:", model.best_estimator_)   #전체 파라미터가 나오고

print("최적의 파라미터:", model.best_params_)      #내가 지정한 것만 나온다

print("best_score:", model.best_score_)

print("model.score:", model.score(x_test, y_test))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠:",r2_score(y_test,y_pred_best)) 

print("걸린시간:", round(end_time-start_time,2), '초')

# 최적의 매개변수: RandomForestRegressor(max_depth=12)
# 최적의 파라미터: {'max_depth': 12, 'n_estimators': 100}
# best_score: 0.6992327181812497
# model.score: 0.8090005616184414
# 최적 튠: 0.8090005616184414
# 걸린시간: 6.48 초