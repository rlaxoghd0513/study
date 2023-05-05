#quantiletranformer 로 스케이링했을때 차이 그림으로 그려서 보기

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['font.family'] = 'Malgun Gothic'
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import QuantileTransformer,PowerTransformer,StandardScaler
#클러스터링: 몇개의 군집으로 만들겠다

x,y = make_blobs(random_state=337,#가우시안 정규분포 샘플 생성
                 centers = 2, #클러스터링할때 기준점을 잡는다 #중심 클러스터 갯수 / y라벨
                 cluster_std=1, #클러스터 표준편차
                 n_samples= 50)  #데이터갯수

print(x)
print(y)
print(x.shape, y.shape) #(50, 2) (50,) 

fig, ax = plt.subplots(2,2,figsize=(12,8))


ax[0,0].scatter(x[:,0],x[:,1],#x의 모든행과 0번째열
            c = y,#y별로 컬러 다르게 
            edgecolors = 'black',#가장가지에 검정색칼러를 넣어라
            ) 
ax[0,0].set_title('오리지날')

scaler = QuantileTransformer(n_quantiles=50) #몇분위로 나눈다
x_trans = scaler.fit_transform(x)

ax[0,1].scatter(x_trans[:,0],x_trans[:,1],#x의 모든행과 0번째열
            c = y,#y별로 컬러 다르게 
            edgecolors = 'black',#가장가지에 검정색칼러를 넣어라
            ) 
ax[0,1].set_title(type(scaler).__name__)

scaler = PowerTransformer() #몇분위로 나눈다
x_trans = scaler.fit_transform(x)

ax[1,0].scatter(x_trans[:,0],x_trans[:,1],#x의 모든행과 0번째열
            c = y,#y별로 컬러 다르게 
            edgecolors = 'black',#가장가지에 검정색칼러를 넣어라
            ) 
ax[1,0].set_title(type(scaler).__name__)

scaler = StandardScaler() #몇분위로 나눈다
x_trans = scaler.fit_transform(x)

ax[1,1].scatter(x_trans[:,0],x_trans[:,1],#x의 모든행과 0번째열
            c = y,#y별로 컬러 다르게 
            edgecolors = 'black',#가장가지에 검정색칼러를 넣어라
            ) 
ax[1,1].set_title(type(scaler).__name__)
plt.show()


