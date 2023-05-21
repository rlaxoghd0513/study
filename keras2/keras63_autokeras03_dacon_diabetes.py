import pandas as pd
import autokeras as ak
from sklearn.model_selection import train_test_split

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
train_csv = train_csv.dropna() 
x = train_csv.drop(['Outcome'],axis =1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# 2. 모델구성
model = ak.StructuredDataClassifier(max_trials=1, overwrite=False)  # 최대 시도 횟수 지정

# 3. 훈련
model.fit(x_train, y_train, epochs=100)

# 4.평가, 결과.
results = model.evaluate(x_test, y_test)
print('결과:', results)

# 결과: [0.5776586532592773, 0.7175572514533997]