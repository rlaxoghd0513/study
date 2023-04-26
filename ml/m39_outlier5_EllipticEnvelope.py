import numpy as np
from sklearn.covariance import EllipticEnvelope

aaa = np.array([[-10, 2, 3, 4, 5, 6, 700, 8, 9, 10, 11, 12, 50],
             [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]])
aaa = np.transpose(aaa)

outliers = EllipticEnvelope(contamination=0.3)

for i in range(aaa.shape[1]):
    outliers.fit(aaa[:, i].reshape(-1, 1))
    results = outliers.predict(aaa[:, i].reshape(-1, 1))
    print(results)

