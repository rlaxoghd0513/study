import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

# Load training and test data
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
train_data['type'] = type_to_HP(train_data['type'])
test_data['type'] = type_to_HP(test_data['type'])

# Select subset of features for One-Class SVM model
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare training data
X_train = train_data[features]

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Apply One-Class SVM
svm = OneClassSVM(kernel='rbf', nu=0.1)
svm.fit(X_train)

# Predict anomalies in test data
test_data_svm = scaler.transform(test_data[features])
svm_predictions = svm.predict(test_data_svm)
svm_predictions = [1 if x == -1 else 0 for x in svm_predictions]

# Add predictions to submission dataframe and save
submission['label'] = pd.DataFrame({'Prediction': svm_predictions})
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission.to_csv(save_path + date + 'submission.csv', index=False)