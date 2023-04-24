import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
#decomposition 분해


#1. 데이터
datasets = load_digits()

x = datasets['data']
y = datasets.target
print(x.shape, y.shape)  #(1797, 64) (1797,)
                    
for i in range(64):
    
    pca = PCA(n_components = 64-i)
#n_components 차원(칼럼) 몇개로 줄일건지
    x = pca.fit_transform(x)


    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=123, shuffle=True)

#2. 모델구성
    model = RandomForestClassifier(random_state=123) #훈련할때마다 바뀌니까 random_state고정

#3. 훈련
    model.fit(x_train, y_train)

#4. 평가,예측
    results = model.score(x_test, y_test)
    print('차원',i,'개 축소:', results)
    
# 차원 0 개 축소: 0.9694444444444444
# 차원 1 개 축소: 0.9666666666666667
# 차원 2 개 축소: 0.9666666666666667
# 차원 3 개 축소: 0.9666666666666667
# 차원 4 개 축소: 0.9694444444444444
# 차원 5 개 축소: 0.9722222222222222
# 차원 6 개 축소: 0.9666666666666667
# 차원 7 개 축소: 0.9722222222222222
# 차원 8 개 축소: 0.9666666666666667
# 차원 9 개 축소: 0.9666666666666667
# 차원 10 개 축소: 0.9722222222222222
# 차원 11 개 축소: 0.9694444444444444
# 차원 12 개 축소: 0.975
# 차원 13 개 축소: 0.975
# 차원 14 개 축소: 0.9722222222222222
# 차원 15 개 축소: 0.9722222222222222
# 차원 16 개 축소: 0.9694444444444444
# 차원 17 개 축소: 0.9694444444444444
# 차원 18 개 축소: 0.9694444444444444
# 차원 19 개 축소: 0.975
# 차원 20 개 축소: 0.9694444444444444
# 차원 21 개 축소: 0.975
# 차원 22 개 축소: 0.9694444444444444
# 차원 23 개 축소: 0.9722222222222222
# 차원 24 개 축소: 0.9722222222222222
# 차원 25 개 축소: 0.9694444444444444
# 차원 26 개 축소: 0.9666666666666667
# 차원 27 개 축소: 0.9722222222222222
# 차원 28 개 축소: 0.9666666666666667
# 차원 29 개 축소: 0.975
# 차원 30 개 축소: 0.9722222222222222
# 차원 31 개 축소: 0.9777777777777777
# 차원 32 개 축소: 0.9666666666666667
# 차원 33 개 축소: 0.975
# 차원 34 개 축소: 0.9694444444444444
# 차원 35 개 축소: 0.9666666666666667
# 차원 36 개 축소: 0.9694444444444444
# 차원 37 개 축소: 0.9666666666666667
# 차원 38 개 축소: 0.9666666666666667
# 차원 39 개 축소: 0.9694444444444444
# 차원 40 개 축소: 0.975
# 차원 41 개 축소: 0.9722222222222222
# 차원 42 개 축소: 0.975
# 차원 43 개 축소: 0.9722222222222222
# 차원 44 개 축소: 0.975
# 차원 45 개 축소: 0.975
# 차원 46 개 축소: 0.9694444444444444
# 차원 47 개 축소: 0.9666666666666667
# 차원 48 개 축소: 0.9666666666666667
# 차원 49 개 축소: 0.9638888888888889
# 차원 50 개 축소: 0.9694444444444444
# 차원 51 개 축소: 0.9611111111111111
# 차원 52 개 축소: 0.9611111111111111
# 차원 53 개 축소: 0.9555555555555556
# 차원 54 개 축소: 0.9583333333333334
# 차원 55 개 축소: 0.9555555555555556
# 차원 56 개 축소: 0.9444444444444444
# 차원 57 개 축소: 0.9472222222222222
# 차원 58 개 축소: 0.9138888888888889
# 차원 59 개 축소: 0.9166666666666666
# 차원 60 개 축소: 0.8472222222222222
# 차원 61 개 축소: 0.7694444444444445
# 차원 62 개 축소: 0.625
# 차원 63 개 축소: 0.32222222222222224