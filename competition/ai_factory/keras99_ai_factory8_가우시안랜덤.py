import pandas as pd
import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import randint, uniform

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
gmm = GaussianMixture(random_state=0)
param_dist = {'n_components': randint(2, 5), 'covariance_type': ['full', 'tied', 'diag', 'spherical'], 'tol': uniform(0.0001, 0.01)}
rs_gmm = RandomizedSearchCV(gmm, param_distributions=param_dist, scoring='f1', cv=5, n_jobs=-1, random_state=0)
rs_gmm.fit(X_train)

# Print best hyperparameters
print("Best hyperparameters:", rs_gmm.best_params_)

# Compute log-likelihood for each sample in train set
log_likelihood_train = rs_gmm.score_samples(X_train)

# Tuning: Adjust the threshold for anomaly detection
threshold = -9.5

# Predict anomalies in test data using trained GMM
test_data_gmm = scaler.transform(test_data[features])
log_likelihood_test = rs_gmm.score_samples(test_data_gmm)
gmm_predictions = [1 if x < threshold else 0 for x in log_likelihood_test]

submission['label'] = pd.DataFrame({'Prediction': gmm_predictions})
print(submission.value_counts())

# Save submission file
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission.to_csv(save_path + date + 'submission.csv', index=False)
