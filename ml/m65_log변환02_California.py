from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#1. 데이터
datasets =  fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names) # x
df['target'] = datasets.target # y
print(df)

# df.boxplot()
# df.plot.box()
# plt.show()

# df.info()
# print(df.describe())

# df['Population'].boxplot()
# #plt.show() -> 얘는 에러뜸
# df['Popuiation'].plot.box()
# plt.show()

# df['Population'].hist(bins=50)
# plt.show()

# df['target'].hist(bins=50)
# plt.show()

y = df['target']
x = df.drop(['target'], axis =1)
print(x.shape,y.shape)

# ################################## x poplation 로그변환 ############################################
x['population'] = np.log1p(x['Population']) # log1p 로그에다가 1을 더하게(plus)해줌.
# #log10의 0은 연산이 안됨.
# #문제점 값이 0이나오면 계산이 안됨. 그래서 연산을 할 수 있게끔 1을 더해줌.

# #지수변환 np.exp1m(마이너스) 지수는 1을 빼주는거.

# ###################### y 로그변환 #############################
y = np.log1p(y)
# ##############################################################

x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.8, shuffle=True,random_state=337)
#분류형은 로그변환이 힘듬

# 2차원 배열로 변환

#2. 모델
model = RandomForestRegressor(random_state=337)
#3. 컴파일, 훈련

model.fit(x_train,y_train)

#4. 평가, 예측
score = model.score(x_test,y_test)

print("r2 :", r2_score(np.expm1(y_test), np.expm1(model.predict(x_test))))

print("score : ", score)