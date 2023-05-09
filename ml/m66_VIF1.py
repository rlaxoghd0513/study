import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor #통계


data = {'size' : [30,35,40,45,50,45],
        'rooms' : [2, 2, 3, 3, 4, 3],
        'window' : [2, 2, 3, 3, 4, 3],
        'year' : [2010,2015,2010,2015,2010,2014],
        'price' : [1.5, 1.8, 2.0, 2.2, 2.5, 2.3]}

df = pd.DataFrame(data)

print(df)
#    size  rooms  year  price
# 0    30      2  2010    1.5
# 1    35      2  2015    1.8
# 2    40      3  2010    2.0
# 3    45      3  2015    2.2
# 4    50      4  2010    2.5
# 5    45      3  2014    2.3

x = df[['size','rooms', 'year', 'window']]
y = df['price']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

print(x_scaled)
# [[-1.61245155 -1.21267813 -0.98994949]
#  [-0.86824314 -1.21267813  1.13137085]
#  [-0.12403473  0.24253563 -0.98994949]
#  [ 0.62017367  0.24253563  1.13137085]
#  [ 1.36438208  1.69774938 -0.98994949]
#  [ 0.62017367  0.24253563  0.70710678]]

vif = pd.DataFrame()
vif['variables'] = x.columns

vif['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])] #x_scaled.shape[1] -> 3개

print(vif)
#   variables         VIF
# 0      size  378.444444
# 1     rooms         inf
# 2      year   53.333333
# 3    window         inf #두개의 컬럼이 완전 똑같아서 지우는게 나음.
#거의 일치
#10이하 일수록 좋음. 높은거부터 제거.

print("========================사이즈 제거전 =============================")
lr = LinearRegression()
lr.fit(x_scaled,y)
y_pred = lr.predict(x_scaled)
r2 = r2_score(y,y_pred)
print("r2 : ", r2)

# ========================사이즈 제거전 =============================
# r2 :  0.9938931297709924

x_scaled = df[['size', 'year', 'window']]

vif2= pd.DataFrame()
vif2['variables'] = x_scaled.columns

vif2['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])] #x_scaled.shape[1] -> 3개
print(vif2)

lr = LinearRegression()
lr.fit(x_scaled,y)
y_pred = lr.predict(x_scaled)
r2 = r2_score(y,y_pred)
print("r2 : ", r2)
# r2 :  0.9938931297709915
#ValueError: Length of values (2) does not match length of index (3) 2개이상은 아니여서.

# ========================사이즈 제거후 =============================
#   variables         VIF
# 0      size  295.182375
# 1      year   56.881874
# 2    window  139.509263