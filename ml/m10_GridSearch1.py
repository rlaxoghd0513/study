import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# 그리드 서치(Grid search)
# 그리드 서치(Grid search)를 통해, 하이퍼파라미터 목록과 성능 지표를 지정하면 알고리즘이 가능한 모든 조합을 통해 작동하여 가장 적합한 것을 결정합니다. 
# 그리드 서치(Grid search)는 훌륭하게 작동하지만 상대적으로 지루하고 계산 집약적입니다. 많은 수의 하이퍼파라미터가 있는 경우 특히 그러합니다.

# 1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=337, test_size=0.2, stratify=y)

gamma = [0.001, 0.01, 0.1, 1, 10, 100]
C = [0.001, 0.01, 0.1, 1, 10, 100]

max_score = 0
for i in gamma:
    for j in C:
        # 2. 모델
        model = SVC(gamma=i, C=j)
        # 3. 컴파일, 훈련
        model.fit(x_train, y_train)

        # 4. 평가, 예측
        score = model.score(x_test, y_test)
        
        if max_score<score:
            max_score=score
            best_parameters = {'gamma': i, 'C':j }

print('max_acc : ', max_score)
print('best_parameters :', best_parameters)