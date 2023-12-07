import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

path = './_data/ai_factory/'
save_path = './_save/ai_factory/'

train_data = pd.read_csv(path + 'train_data.csv')
test_data = pd.read_csv(path + 'test_data.csv')
submission = pd.read_csv(path + 'answer_sample.csv')

data = pd.concat([train_data, test_data], axis=0)
y = []

for index, row in data.iterrows():
    if row['type'] == 1:
        y.append(20)
    elif row['type'] ==2:
        y.append(10)
    elif row['type'] ==3:
        y.append(50)
    elif row['type'] in [0,4,5,6,7]:
        y.append(30)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
print(data.shape)#(9852, 8)

model = DBSCAN(eps = 0.5, min_samples = 5)
model.fit(scaled_data)

# 클러스터 레이블 가져오기
labels = model.labels_

# 클러스터 레이블과 이상치 여부를 확인하기 위해 데이터 프레임에 추가.
data['cluster_label'] = labels
data['is_outlier'] = labels == -1

# 이상치 데이터 출력
outliers = data[data['is_outlier']]
print(outliers)