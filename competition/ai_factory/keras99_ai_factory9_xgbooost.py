import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import xgboost as xgb

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

# Select subset of features for XGBoost model
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and test data
X = train_data[features]
y = pd.Series([0]*len(train_data)) # Assign all labels to 0 as all data are normal

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size= 0.9, random_state= 5555)

# Normalize data
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train XGBoost model
model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=5555)
model.fit(X_train, y_train)

# Predict anomalies in test data using trained XGBoost model
test_data_xgb = scaler.transform(test_data[features])
xgb_predictions = model.predict(test_data_xgb)

submission['label'] = pd.DataFrame({'Prediction': xgb_predictions})
print(submission.value_counts())

# Save submission file
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission.to_csv(save_path + date + 'submission.csv', index=False)