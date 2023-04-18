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
kfold = StratifiedKFold(n_splits = n_splits, shuffle= True, random_state=3654)


parameters = [                                                     #리스트 안에 딕셔너리 형태
    {"C":[1,10,100,1000], "kernel":['linear'],'degree':[3,4,5]},   #12번 돈다    약간씩 자기가 하고 싶은 대로 수정 가능한것
    {"C":[1,10,100],'kernel':['rbf'],'gamma':[0.001,0.0001]},      #12번 돈다
    {"C":[1,10,100,1000], 'kernel':['sigmoid'],                    #24번 돈다
     'gamma':[0.01,0.001,0.0001], 'degree':[3,4]},
    {"C":[0.1,1], "gamma":[1,10]}                               #6번 돈다
    ]                                                              #총 54번 돈다

#모델구성
model = GridSearchCV(SVC(), parameters,
                    #  cv=kfold,     #cv=5도 가능   kfold에 stratify가 안들어가있는데 5로 하니까 더 좋아졌다 디폴트로 stratify가 들어가 있단 얘기
                     cv=5,            #분류의 디폴트는 StratifiedKFold이다 
                     verbose=1, refit=True, n_jobs=-1) #gridsearch1에서 for문으로 길게 돌린게 한줄로 정리된다  kfold했기때문에 5번 더 돌아서 240번 돈다

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



        
