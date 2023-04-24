import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
# print(datasets.feature_names)  #판다스는 columns
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets['data']
y = datasets.target

df = pd.DataFrame(x,columns=datasets.feature_names) #넘파이를 판다스로 바꾸고 칼럼 이름 넣기
print(df)
#[150 rows x 4 columns]

df['Target(Y)'] = y   #Target(Y)를 만들고 y를 넣겠다
print(df)
#[150 rows x 5 columns]

print('=====================상관계수 히트 맵=====================')
print(df.corr()) #Correlation 상관    

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()
#상관관계가 높은 x변수 삭제 고려해본다
#상관관계가 높은 y변수는 좋다
