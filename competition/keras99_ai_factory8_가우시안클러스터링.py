import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# 데이터 로드
path='./_data/ai_factory/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Feature Engineering
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)

def calculate_power(data):
    data['power'] = data['motor_current'] * data['motor_rpm'] / 1000
    return data

def clustering_features(data, n_clusters):
    features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']
    cluster_data = data[features]
    scaler = RobustScaler()
    cluster_data = scaler.fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(cluster_data)
    cluster_labels = kmeans.labels_
    for i in range(n_clusters):
        data[f'cluster_{i+1}'] = (cluster_labels == i).astype(int)
    return data

train_data['type'] = type_to_HP(train_data['type'])
test_data['type'] = type_to_HP(test_data['type'])
train_data = calculate_power(train_data)
test_data = calculate_power(test_data)
train_data = clustering_features(train_data, n_clusters=3)
test_data = clustering_features(test_data, n_clusters=3)

# Prepare train and test data
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe', 'power', 'cluster_1', 'cluster_2', 'cluster_3']
X_train = train_data[features].drop('Time', axis=1)
X_test = test_data[features].drop('Time', axis=1)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
gmm.fit(X_train)

# Predict outliers for test data
y_pred = gmm.predict_proba(X_test)[:, 2]
threshold = np.percentile(y_pred, 5)
outliers = (y_pred < threshold).astype(int)

# Save submission file
submission['label'] = outliers
submission.to_csv(path+'submission.csv', index=False)