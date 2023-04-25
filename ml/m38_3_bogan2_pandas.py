import numpy as np
import pandas as pd

data = pd.DataFrame([[2,np.nan,6,8,10],
                    [2,4,np.nan,8,np.nan],
                    [2,4,6,8,10],
                    [np.nan,4,np.nan,8,np.nan]]).transpose()

print(data)
data.columns = ['x1','x2','x3','x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

#0 결측치 확인
print(data.isnull())  #True가 결측치란 얘기
print(data.isnull().sum())
# x1    1
# x2    2
# x3    0
# x4    3
####데이터받으면 info랑 describe봐라 
print(data.info())
print(data.describe())

#2 결측치 삭제
print('====결측치삭제====')
print(data['x1'].dropna()) #이렇게 하면 그 열에서만 삭제된다
print(data.dropna()) #디폴트가 행 위주로 삭제
print(data.dropna(axis=0))
print(data.dropna(axis=1))

#2-1 특정값 - 평균
print('================결측치처리 mean()===============')
means = data.mean()
print('평균:', means)
data2 = data.fillna(means)
print(data2)

#2-2 특정값 - 중위값
print('==============결측치처리 median()===============')
median = data.median()
print('중위값:', median)
data3 = data.fillna(median)
print(data3)

#2-3 0fillna
print('================== 결측치처리 0===============')
data4 = data.fillna(0)
print(data4)

#2-4 앞에값으로채우기
print('====================결측치처리 ffill===============')
data5 = data.ffill()
# data5 = data.fillna(method = 'ffill')
print(data5)

#2-5 뒤에값으로 채우기
print('===================결측치처리 bfill===============')
data6 = data.bfill()
# data6 = data.fillna(method='backfill')
print(data6)

#2-6 임의의값으로 채우기
print('====================결측치처리 임의의 값으로 채우기======================')
data7 = data.fillna(777777777)
# data7 = data.fillna(value=77777)
print(data7)

#3 보간
print('================== 결측치처리 bogan==============')
data7 = data.interpolate()
print(data7)

##################################특정칼럼만!!!####################################
#1. x1칼럼에 평균값
data['x1'] = data['x1'].fillna(data['x1'].mean())
print(data['x1'])

#2. x2칼럼에 중위값
data['x2'] = data['x2'].fillna(data['x2'].median())
print(data['x2'])

#3. x4칼럼에 ffill한 후 제일 위에 남은 행에 77777채우기
# data['x4'] = data['x4'].ffill()

# data['x4'][0] = '7777777'
# data['x4'] = data['x4'].fillna(7777777)

data['x4'] = data['x4'].fillna(method='ffill').fillna(value=77777777)

print(data['x4'])
print(data)








































