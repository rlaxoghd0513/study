#그래프 그리기
#1. value_counts -> 금지
#2. np.unique의 return_counts -> 금지
#3. groupby 쓴다. count
# plt.bar 로 그린다 (quality 컬럼)
# 데이터 개수(y축) = 데이터 갯수, 


import pandas as pd
import matplotlib.pyplot as plt

#.1 데이터
path='./_data/dacon_wine/'
data_set=pd.read_csv(path+'train.csv',index_col=0)
# print(data_set.shape) (4898, 11)                                ㄴ기준으로 컬럼을 나눠줘

# print(data_set.describe())
# print(data_set.info())

x=data_set.drop(['quality'],axis=1)
y=data_set['quality']

#########그래프 그리기#########
# 1. value_counts 쓰지마라
# 2. groupby , count() 써보기
# plt.bar (quality)

data_counts = data_set.groupby('quality')['quality'].count()
#                           퀄리티로 묵겠다,퀄리티지정,카운트하겠다.
plt.bar(data_counts.index, data_counts)
plt.xlabel('quality')
plt.ylabel('count')
plt.grid()
# plt.legend('legend')
plt.show()
print(data_counts)
