import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import autokeras as ak
#1. 데이터

path = './_data/wine/'
path_save = './_save/wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

le = LabelEncoder()
le.fit(train_csv['type'])
train_csv['type'] = le.transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']

# 1-1 train test 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 2. 모델구성
model = ak.StructuredDataClassifier(max_trials=1, overwrite=False)  # 최대 시도 횟수 지정

# 3. 훈련
model.fit(x_train, y_train, epochs=100)

# 4.평가, 결과.
results = model.evaluate(x_test, y_test)
# print('결과:', results) 결과: [1.1407880783081055, 0.5454545617103577]