import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.mixture import GaussianMixture

# 훈련 데이터 및 테스트 데이터 로드
path='./_data/ai_factory/'
save_path= './_save/ai_factory/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Preprocess data
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])

# Select subset of features for GMM model
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and test data
X = train_data[features]

# Split data into train and validation sets
X_train, X_val = train_test_split(X, train_size= 0.9, random_state= 5555)

# Normalize data
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
gmm.fit(X_train)

# Compute log-likelihood for each sample in train set
log_likelihood_train = gmm.score_samples(X_train)

import matplotlib.pyplot as plt
import numpy as np

# Compute log-likelihood for each sample in validation set
log_likelihood_val = gmm.score_samples(X_val)

# Visualize distribution of log-likelihoods for validation set
plt.hist(log_likelihood_val, bins=50, density=True, alpha=0.5, color='blue')

# Compute threshold based on percentile of log-likelihoods
threshold = np.percentile(log_likelihood_val, 5)
print("Threshold:", threshold)

# Plot threshold
plt.axvline(x=threshold, color='red')
plt.show()