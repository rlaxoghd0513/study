import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
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
X_train = train_data[features]
X_test = test_data[features]

# Normalize data
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Split data into train and validation sets
X_train_pca, X_val_pca = train_test_split(X_train_pca, train_size= 0.9, random_state= 5555)

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
gmm.fit(X_train_pca)

# Compute log-likelihood for each sample in train set
log_likelihood_train = gmm.score_samples(X_train_pca)

# Tuning: Adjust the threshold for anomaly detection
threshold = -9.5

# Predict anomalies in test data using trained GMM
log_likelihood_test = gmm.score_samples(X_test_pca)
gmm_predictions = [1 if x < threshold else 0 for x in log_likelihood_test]

submission['label'] = pd.DataFrame({'Prediction': gmm_predictions})
print(submission.value_counts())

# Save submission file
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission.to_csv(save_path + date + 'submission.csv', index=False)