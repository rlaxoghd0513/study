from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#1. 데이터
datasets =  load_diabetes()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names) # x
df['target'] = datasets.target # y

# df.plot.box()
# plt.show()

# df.info()
# print(df.describe())

# df['target'].plot.box()
# plt.show()

y = df['target']
x = df.drop(['target'],axis = 1)

####################로그변환############################
x['age'] = np.log1p(x['age']) 
y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.8, shuffle=True,random_state=337)

y_train_log = np.log1p(y_train) 
y_test_log = np.log1p(y_test) 

#2. 모델
model = RandomForestRegressor(random_state=337)
#3. 컴파일, 훈련

model.fit(x_train,y_train)

#4. 평가, 예측
score = model.score(x_test,y_test)

print("로그-> 지수 :", r2_score(np.expm1(y_test), np.expm1(model.predict(x_test))))

print("score : ", score)

