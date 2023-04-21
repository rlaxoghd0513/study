import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

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

# Select subset of features for Isolation Forest model
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and test data
X = train_data[features]

# Split data into train and validation sets
X_train, X_val = train_test_split(X, train_size= 0.9, random_state= 5555)

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Apply Isolation Forest
iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=5555)
y_pred_train = iso.fit_predict(X_train)

# Tuning: Adjust the n_estimators and contamination parameters
iso_tuned = IsolationForest(n_estimators=200, contamination=0.048, random_state=5555)
y_pred_train_tuned = iso_tuned.fit_predict(X_train)

# Predict anomalies in test data using tuned Isolation Forest
test_data_iso = scaler.transform(test_data[features])
y_pred_test_iso = iso_tuned.fit_predict(test_data_iso)
iso_predictions = [1 if x == -1 else 0 for x in y_pred_test_iso]

submission['label'] = pd.DataFrame({'Prediction': iso_predictions})
print(submission.value_counts())
#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(save_path + date + 'submission.csv', index=False)