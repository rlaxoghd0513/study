import numpy as np
import pandas as pd
from datetime import datetime

dates = ['4/25/2023','4/26/2023','4/27/2023','4/28/2023','4/29/2023','4/30/2023']

dates = pd.to_datetime(dates)
print(dates)
print(type(dates))#<class 'pandas.core.indexes.datetimes.DatetimeIndex'>

print('===========================================')
ts = pd.Series([2,np.nan,np.nan,8,10,np.nan], index = dates)
#판다스안에 있는 데이터는 어차피 넘파이 형태다 

print(ts)

# 스칼라 데이터 한개   모이면 벡터 1차원   벡터모이면 행렬 매트릭스    매트릭스 모이면 텐서 
# 판다스 한개짜리 데이터들이 모이면 시리즈 (벡터와 같다)   시리즈 모이면 데이터프레임 

print('============================')
ts = ts.interpolate() #판다스에서 제공한다  결측치 채우는거 
print(ts)
