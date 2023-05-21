import autokeras as ak
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

# 1. 데이터
data = fetch_covtype()
x = data.data
y = data.target

# 1-1 train test 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 2. 모델구성
model = ak.StructuredDataClassifier(max_trials=1, overwrite=False)  # 최대 시도 횟수 지정

# 3. 훈련
model.fit(x_train, y_train, epochs=100)

# 4.평가, 결과.
results = model.evaluate(x_test, y_test)
print('결과:', results)