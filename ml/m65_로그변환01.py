import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=2.0, size=1000)

#로그변환
log_data = np.log(data) #log는 제곱수를 구하는 것.

#원본 데이터 히스토그램 그리자

plt.subplot(1, 2, 1)
plt.hist(data, bins= 50, color='blue', alpha = 0.5)
#plt.titie('Original')

plt.subplot(1, 2, 2)
plt.hist(log_data, bins= 50, color='red', alpha = 0.5)
#plt.titie('Log Transforms Data')

plt.show()

#로그데이터의 효과 데이터가 가운데로 몰림. -> 