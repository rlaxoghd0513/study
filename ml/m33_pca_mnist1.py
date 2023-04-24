import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()   #y가져오기 싫다 파이썬 기초문법 _작대기
# (x_train, _), (_, _) = mnist.load_data()#x_test도 가져오기 싫다

# x = np.concatenate([x_train, x_test], axis=0)
x = np.append(x_train, x_test, axis=0)
print(x.shape) #(70000, 28, 28)

#실습
#pca를 통해 0.95 이상인 n_components 는 몇개?
#0.95몇개
#0.99몇개
#0.999몇개
#1.0 몇개
x = x.reshape(70000,784)

pca = PCA(n_components = 784)
x = pca.fit_transform(x)
print(x.shape)           #(569, 30)

pca_EVR = pca.explained_variance_ratio_ #설명가능한 변화율
print(pca_EVR)
print(sum(pca_EVR)) #0.9999999999999998  순서대로 pca1개했을때 2개했을때 3개 했을때---

print(np.cumsum(pca_EVR)) #누적합 pca를 몇개하면 데이터에 손실이 없을지 보는거 1부분에 가까울수록 데이터 손실이 없다
pca_cumsum = np.cumsum(pca_EVR)

import matplotlib.pyplot as plt
plt.plot(pca_cumsum)
plt.grid()
plt.show()

print(np.argmax(pca_cumsum >= 0.95)+1) #154
print(np.argmax(pca_cumsum >= 0.99)+1) #331
print(np.argmax(pca_cumsum >= 0.999)+1) #486
print(np.argmax(pca_cumsum >= 1.0)+1) #713






