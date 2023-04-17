import pandas as pd
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor

# 훈련 데이터 및 테스트 데이터 로드
path = './_data/ai_factory/'
save_path = './_save/ai_factory/'
train_data = pd.read_csv(path + 'train_data.csv')
test_data = pd.read_csv(path + 'test_data.csv')
submission = pd.read_csv(path + 'answer_sample.csv')

# Preprocess data
def type_to_HP(type):
    HP = [30, 20, 10, 50, 30, 30, 30, 30]
    gen = (HP[i] for i in type)
    return list(gen)

train_data['type'] = type_to_HP(train_data['type'])
test_data['type'] = type_to_HP(test_data['type'])

# Select subset of features for LOF model
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and test data
X = train_data[features]

# Split data into train and validation sets
X_train, X_val = train_test_split(X, train_size=0.9, random_state=3764)

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define parameter distribution for random search
param_dist = {
    "n_neighbors": np.arange(5, 51),
    "contamination": np.random.uniform(0.01, 0.2, 1000)
}

# 비지도 학습을 위해 LocalOutlierFactor 객체 생성
lof = LocalOutlierFactor()

# 가장 좋은 하이퍼파라미터를 찾기 위해 랜덤 서치를 구현합니다.
best_score = -np.inf
best_params = None

for _ in range(50):
    n_neighbors = np.random.choice(param_dist["n_neighbors"])
    contamination = np.random.choice(param_dist["contamination"])
    
    lof.set_params(n_neighbors=n_neighbors, contamination=contamination)
    y_pred_train = lof.fit_predict(X_train)
    score = np.sum(y_pred_train == 1) / len(y_pred_train)
    
    if score > best_score:
        best_score = score
        best_params = {"n_neighbors": n_neighbors, "contamination": contamination}

print("Best parameters: ", best_params)

# Train LOF with the best parameters
lof_tuned = LocalOutlierFactor(n_neighbors=best_params['n_neighbors'], contamination=best_params['contamination'])
y_pred_train_tuned = lof_tuned.fit_predict(X_train)

# Predict anomalies in test data using tuned LOF
test_data_lof = scaler.transform(test_data[features])
y_pred_test_lof = lof_tuned.fit_predict(test_data_lof)
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]

submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
print(submission.value_counts())
#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(save_path + date + 'submission.csv', index=False)