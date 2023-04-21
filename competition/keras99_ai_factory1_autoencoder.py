import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 데이터 로드 및 전처리
path='./_data/ai_factory/'
save_path= './_save/ai_factory/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)

train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])

features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data[features])
test_data_scaled = scaler.transform(test_data[features])

# Autoencoder 모델 정의
input_layer = Input(shape=(7,))
encoder_layer1 = Dense(5, activation='relu')(input_layer)
encoder_layer2 = Dense(3, activation='relu')(encoder_layer1)
decoder_layer1 = Dense(5, activation='relu')(encoder_layer2)
output_layer = Dense(7, activation='sigmoid')(decoder_layer1)
model = Model(input_layer, output_layer)
model.compile(optimizer='adam', loss='mse')

# 모델 학습
model.fit(train_data_scaled, train_data_scaled, epochs=50, batch_size=32, validation_split=0.1)

# 재구성 오차 계산
train_preds = model.predict(train_data_scaled)
train_mses = np.mean(np.power(train_data_scaled - train_preds, 2), axis=1)

# 재구성 오차 기준으로 이상치 탐지
threshold = np.percentile(train_mses, 95)
test_preds = model.predict(test_data_scaled)
test_mses = np.mean(np.power(test_data_scaled - test_preds, 2), axis=1)
anomalies = test_mses > threshold

# 결과 저장
lof_predictions = [1 if x else 0 for x in anomalies]
submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission.to_csv(save_path + date + 'submission.csv', index=False)