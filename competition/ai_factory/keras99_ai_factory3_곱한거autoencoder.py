import pandas as pd
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics import mean_squared_error

# Load train and test data
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

# Create a new feature by multiplying motor_current and motor_rpm
train_data['motor_current_rpm'] = train_data['motor_current'] * train_data['motor_rpm']
test_data['motor_current_rpm'] = test_data['motor_current'] * test_data['motor_rpm']

# Select subset of features for autoencoder model
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_temp', 'motor_vibe', 'motor_current_rpm']

# Prepare train and test data
X_train = train_data[features].values
X_test = test_data[features].values

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define autoencoder architecture
input_dim = X_train.shape[1]
encoding_dim = 3
input_layer = Input(shape=(input_dim, ))
encoder_layer1 = Dense(64, activation="relu")(input_layer)
encoder_layer2 = Dense(32, activation="relu")(encoder_layer1)
encoder_layer3 = Dense(encoding_dim, activation="relu")(encoder_layer2)
decoder_layer1 = Dense(32, activation="relu")(encoder_layer3)
decoder_layer2 = Dense(64, activation="relu")(decoder_layer1)
output_layer = Dense(input_dim, activation="sigmoid")(decoder_layer2)

autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Compile and fit the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 50, restore_best_weights=True)
autoencoder.fit(X_train, X_train, epochs=1000, batch_size=32, shuffle=True, validation_data=(X_test, X_test), callbacks= [es])

# Predict anomalies in test data using the trained autoencoder
train_pred = autoencoder.predict(X_train)
mse_train = np.mean(np.power(X_train - train_pred, 2), axis=1)
test_pred = autoencoder.predict(X_test)
mse_test = np.mean(np.power(X_test - test_pred, 2), axis=1)

# Set the threshold for anomaly detection
threshold = np.percentile(mse_train, 95)

# Predict anomalies in test data using the threshold
autoencoder_predictions = [1 if mse > threshold else 0 for mse in mse_test]

submission['label'] = pd.DataFrame({'Prediction': autoencoder_predictions})
print(submission.value_counts())

# Save submission
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission.to_csv(save_path + date + 'submission.csv', index=False)