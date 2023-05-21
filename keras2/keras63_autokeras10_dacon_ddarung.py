import autokeras as ak
from sklearn.model_selection import train_test_split
import pandas as pd

#1 데이터
path = './_data/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

# 1-1 train test 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 2. 모델구성
model = ak.StructuredDataRegressor(max_trials=1, overwrite=False)  # 최대 시도 횟수 지정

# 3. 훈련
model.fit(x_train, y_train, epochs=100)

# 4.평가, 결과.
results = model.evaluate(x_test, y_test)
print('결과:', results) #결과: [3541.418212890625, 3541.418212890625]