#칼럼의 갯수가 클래스의 갯수보다 작을때
#디폴트로 돌아가나


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from tensorflow.keras.datasets import cifar100

#1 데이터
(x_train,y_train),(x_test, y_test) = cifar100.load_data()
print(x_train.shape) #(50000, 32, 32, 3)

x_train = x_train.reshape(50000,32*32*3)
print(x_train.shape) #(50000, 3072)

pca = PCA(n_components=99) 
x = pca.fit_transform(x_train)
print(x.shape) #(50000, 99)

lda = LinearDiscriminantAnalysis() #n_components 디폴트: 전체클래스갯수 -1  #n_components 가 전체클래스보다 작게 넣어야한다
#lda는 클래스별로 매치시킨다했기 때문에 fit_transform할때 y값이 필요하다
#pca가 좋은지 lda가 좋은지는 해봐야 알 수 있다

x = lda.fit_transform(x_train,y_train)
print(x.shape) #(50000,99)
