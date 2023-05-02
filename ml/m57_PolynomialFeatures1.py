# x만 polynomial 
# 데이터를 다항식 형태로 변환

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


x = np.arange(8).reshape(4,2)
print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
print(x.shape)#(4, 2)

print('==================degree2===========================')

poly = PolynomialFeatures(degree=2) #칼럼 증폭
x = poly.fit_transform(x)
print(x)
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]
#   1 원  원  제곱  곱  제곱
#poly한거 첫번째 칼럼은 무조건 1들어간다

print(x.shape)#(4, 6)

print('====================degree3====================================')

x = np.arange(8).reshape(4,2)

poly = PolynomialFeatures(degree=3) #칼럼 증폭
x = poly.fit_transform(x)
print(x)
# [[  1.   0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  1.   2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  1.   4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  1.   6.   7.  36.  42.  49. 216. 252. 294. 343.]]
                                #세제곱  2열제곱*3열  3열제곱*2열 세제곱

print(x.shape)#(4, 10)

print('==========================칼럼3 degree2==============================')

x = np.arange(12).reshape(4,3)
print(x)

# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

poly = PolynomialFeatures(degree=2) #칼럼 증폭
x = poly.fit_transform(x)
print(x)
# [[  1.   0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  1.   3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  1.   6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  1.   9.  10.  11.  81.  90.  99. 100. 110. 121.]]

print(x.shape)#(4, 10)
