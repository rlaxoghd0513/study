import numpy as np
aaa = np.array([-10,2,3,4,5,6,700,8,9,10,11,12,50])

aaa = aaa.reshape(-1,1)

from sklearn.covariance import EllipticEnvelope #2차원만 받는다
outlier = EllipticEnvelope(contamination=.2) #contamination 전체 데이터중에 몇프로를 이상치로 판단할거냐

outlier.fit(aaa)
results = outlier.predict(aaa)
print(results)
