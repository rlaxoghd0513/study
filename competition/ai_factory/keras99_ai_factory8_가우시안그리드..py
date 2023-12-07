import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
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

# Split data into train and validation sets.
X_train, X_val = train_test_split(X, train_size= 0.9, random_state= 1234)

n_splits = 5
kfold = KFold(n_splits = n_splits, random_state = 1234, shuffle=True)

parameters = [
    {'n_components':[1,2,3,4], 'covariance_type':['spherical','diag','tied','full'],
     'max_iter':[100,200,300,400,500,1000],'n_init':[1,2,3]}
]

# Normalize data
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Apply Gaussian Mixture Model
gmm = GridSearchCV(GaussianMixture(), parameters, cv=kfold, n_jobs=-1)
gmm.fit(X_train)

# Compute log-likelihood for each sample in train set
log_likelihood_train = gmm.score_samples(X_train)

# Compute threshold value based on log-likelihood scores of training set
threshold = -9.5

# Predict anomalies in test data using trained GMM
test_data_gmm = scaler.transform(test_data[features])
log_likelihood_test = gmm.score_samples(test_data_gmm)
gmm_predictions = [1 if x < threshold else 0 for x in log_likelihood_test]

submission['label'] = pd.DataFrame({'Prediction': gmm_predictions})
print(submission.value_counts())

# Save submission file
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission.to_csv(save_path + date + 'submission.csv', index=False)
