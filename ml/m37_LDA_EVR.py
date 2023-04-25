import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype
from tensorflow.keras.datasets import cifar100

#1 데이터
data_list = [load_iris, load_wine, load_digits, load_breast_cancer, fetch_covtype]
data_list_name = ['iris', 'wine', 'digits', 'cancer', 'covtype']

for i,value in enumerate(data_list):
    
    x,y = value(return_X_y=True)
    lda = LinearDiscriminantAnalysis()
    x_lda = lda.fit_transform(x,y)
    print('=======================')
    print(data_list_name[i])
    print(x.shape,'->',x_lda.shape)
    lda_EVR = lda.explained_variance_ratio_
    cumsum = np.cumsum(lda_EVR)
    print(cumsum)
    
# =======================
# iris
# (150, 4) -> (150, 2)
# [0.9912126 1.       ]
# =======================
# wine
# (178, 13) -> (178, 2)
# [0.68747889 1.        ]
# =======================
# digits
# (1797, 64) -> (1797, 9)
# [0.28912041 0.47174829 0.64137175 0.75807724 0.84108978 0.90674662
#  0.94984789 0.9791736  1.        ]
# =======================
# cancer
# (569, 30) -> (569, 1)
# [1.]
# =======================
# covtype
# (581012, 54) -> (581012, 6)
# [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]

