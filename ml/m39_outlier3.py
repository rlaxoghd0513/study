import numpy as np
import matplotlib.pyplot as plt

aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-700000,800,900,1000,210,420,350]])

aaa = np.transpose(aaa)
print(aaa.shape) #(13, 2)
print(aaa)

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25,50,75], axis=0)
    print('1사분위:', quartile_1)
    print('q2:',q2)
    print('3사분위:', quartile_3)
    iqr = quartile_3 - quartile_1
    print('iqr:', iqr)
    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 + (iqr*1.5)
    outlier_indices = np.where((data_out>upper_bound)|(data_out<lower_bound))
    outlier_values = data_out[outlier_indices]
    return outlier_indices, outlier_values

outliers_loc, outliers_val = outliers(aaa)
print('이상치의 위치:', np.transpose(outliers_loc))
print('이상치의 값:', outliers_val)

plt.boxplot(aaa)
plt.show()

# [[    -10     100]
#  [      2     200]
#  [      3     -30]
#  [      4     400]
#  [      5     500]
#  [      6     600]
#  [      7 -700000]
#  [      8     800]
#  [      9     900]
#  [     10    1000]
#  [     11     210]
#  [     12     420]
#  [     50     350]]

# 1사분위: [  4. 200.]
# q2: [  7. 400.]
# 3사분위: [ 10. 600.]
# iqr: [  6. 400.]
# 이상치의 위치: [[ 0  0]
#                [ 6  1]
#                [12  0]]
# 이상치의 값: [    -10 -700000      50]