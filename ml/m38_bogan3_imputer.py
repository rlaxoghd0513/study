import numpy as np
import pandas as pd
import sklearn as sk
print(sk.__version__) #1.2.2

data = pd.DataFrame([[2,np.nan,6,8,10],
                    [2,4,np.nan,8,np.nan],
                    [2,4,6,8,10],
                    [np.nan,4,np.nan,8,np.nan]]).transpose()

data.columns = ['x1','x2','x3','x4']
# print(data)
    #   0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer  #impute 책임을 전가하다 -> 결측치에 대한 책임을 돌리다
from sklearn.impute import KNNImputer #최근접 이웃값에 상정하는 거 
from sklearn.impute import IterativeImputer #
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# imputer = SimpleImputer() #디폴트 평균값 == 결측치 채우기
# imputer = SimpleImputer(strategy = 'median') #중위값
# imputer = SimpleImputer(strategy = 'most_frequent') #가장 자주 등장한 값 -> 빈도수가 동일하면 제일 작은 값으로 결측치 채운다
# imputer = SimpleImputer(strategy = 'constant',fill_value=7777)  #특정값 넣는다
# imputer = KNNImputer() #디폴트 평균값
imputer = IterativeImputer(estimator=XGBRegressor()) #????
data2 = imputer.fit_transform(data) #결측치에 각 열의 평균을 넣는다
print(data2)



