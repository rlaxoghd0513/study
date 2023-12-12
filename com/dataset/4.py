import pandas as pd
from sklearn.model_selection import train_test_split

#데이터 전처리

path = './com/dataset/'
path_save= './com/dataset/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# print(train_csv) #[43865 rows x 35 columns]
# print(test_csv) #[18798 rows x 35 columns]
# print(train_csv.info()) #결측치 없음

###########################################################################################################
#시각화

# import seaborn as sns
# import matplotlib.pyplot as plt

# # X1부터 X30까지의 칼럼과 A, B, C, D, E 칼럼 선택
# selected_columns = train_csv.iloc[:, :30].join(train_csv.iloc[:, -5:])

# # 상관 행렬 계산
# correlation_matrix = selected_columns.corr()

# # 히트맵 시각화
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix.iloc[:30, -5:], annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap between X1-X30 and A, B, C, D, E columns')
# plt.show()

# # A, B, C, D, E 칼럼 선택
# selected_columns = train_csv[['A', 'B', 'C', 'D', 'E']]

# # A, B, C, D, E 간의 상관 행렬 계산
# correlation_matrix = selected_columns.corr()

# # 히트맵 시각화
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap between A, B, C, D, E columns')
# plt.show()



# # # 각 열의 분포 시각화 (히스토그램)
# for column in train_csv.columns:
#     sns.histplot(train_csv[column], kde=True)  # 히스토그램 생성
#     plt.title(f'Distribution of Column: {column}')
#     plt.xlabel('Values')
#     plt.ylabel('Frequency')
#     plt.show()
    
#한지점에 모인 칼럼 1,6,9,,10,
#여러군데로 분포된 칼럼 2,3,7,8,15 16 17 18 21 23 
#두군대로 모인 칼럼 4,11,12,13 19 20 22 25 27 30
#고르게 분포된 칼럼 5,14,24 26 28 29 

##################################################################################################################
# 낮은 상관관계 제거할 열
cols_to_drop = ['X1','X5', 'X10', 'X14']

train_csv = train_csv.drop(cols_to_drop, axis=1)
test_csv = test_csv.drop(cols_to_drop, axis=1)

# print(train_csv)

##################################################################################################################
#x,y지정
x = train_csv.iloc[:, :26]  
y = train_csv.iloc[:, 26:] 
# print(x.shape, y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=1336, shuffle=True)

#####################################################################################################################
#스케일링
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler
min_scaler = MinMaxScaler()
rob_scaler = RobustScaler()
max_scaler = MaxAbsScaler()
sta_scaler = StandardScaler()

scale_min_columns = ['X4','X6','X9','X11','X12','X13','X19','X20','X22','X25','X27','X30']

x_train[scale_min_columns] \
    = min_scaler.fit_transform(x_train[scale_min_columns])

x_test[scale_min_columns] \
    = min_scaler.transform(x_test[scale_min_columns])
    
test_csv[scale_min_columns] \
    = min_scaler.transform(test_csv[scale_min_columns])
    
####################################################

scale_rob_columns = ['X2','X3','X7','X8','X15','X16','X17','X18','X21','X23','X24','X26','X28','X29']

x_train[scale_rob_columns] \
    = rob_scaler.fit_transform(x_train[scale_rob_columns])

x_test[scale_rob_columns] \
    = rob_scaler.transform(x_test[scale_rob_columns])
    
test_csv[scale_rob_columns] \
    = rob_scaler.transform(test_csv[scale_rob_columns])
    
# print(x_train)#[35092 rows x 26 columns]
# print(x_test)#[8773 rows x 26 columns]
# print(test_csv)#[18798 rows x 31 columns]
    
##################################################################################################################################
#모델구성


from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import RandomizedSearchCV

param_grid_rf = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 각 출력에 대해 별도의 모델 생성
model_A = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=param_grid_rf, n_iter=100, cv=3, verbose=1, random_state=1336, n_jobs=-1)
model_B = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=param_grid_rf, n_iter=100, cv=3, verbose=1, random_state=1336, n_jobs=-1)
model_C = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=param_grid_rf, n_iter=100, cv=3, verbose=1, random_state=1336, n_jobs=-1)
model_D = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=param_grid_rf, n_iter=100, cv=3, verbose=1, random_state=1336, n_jobs=-1)
model_E = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=param_grid_rf, n_iter=100, cv=3, verbose=1, random_state=1336, n_jobs=-1)

# 다중 출력 회귀 모델에 각 모델을 넣어 모델 생성
multi_output_model = MultiOutputRegressor(estimator=RandomForestRegressor())

# 각 모델을 다중 출력 모델에 할당
multi_output_model.estimators_ = [model_A, model_B, model_C, model_D, model_E]

# 모델 학습
multi_output_model.fit(x_train, y_train)

# 테스트 데이터에 대한 예측
predictions = multi_output_model.predict(x_test)

# MAE 계산
mae = mean_absolute_error(y_test, predictions)
print('mae:', mae)

x_predict = test_csv.iloc[:, :26]  
y_predict = test_csv.iloc[:, 26:]

predict_test_csv = multi_output_model.predict(x_predict)
mae_predict = mean_absolute_error(y_predict, predict_test_csv)
print('mae_test_csv:', mae_predict)

###############################################################################################
from collections import Counter

# 각 열에서 가장 많이 등장하는 값을 찾기
most_common_values = {
    'A': Counter(y_train['A']).most_common(1)[0][0],
    'B': Counter(y_train['B']).most_common(1)[0][0],
    'C': Counter(y_train['C']).most_common(1)[0][0],
    'D': Counter(y_train['D']).most_common(1)[0][0],
    'E': Counter(y_train['E']).most_common(1)[0][0]
}

# 모델이 예측한 값
predicted_values = multi_output_model.predict(x_predict)

better_counter = 0  
rows_with_3_better = []  

# 각 행에 대해 조건 검사 후 출력
for index, row in enumerate(predicted_values):
    better_count = 0  # 'better'의 개수를 위한 변수 초기화
    for i, value in enumerate(row):
        # ABC는 값이 높으면 better, 낮으면 worse
        if i < 3:
            if value >= most_common_values[chr(65 + i)]:
                better_count += 1
        # DE는 값이 낮으면 better, 높으면 worse
        else:
            if value <= most_common_values[chr(65 + i)]:
                better_count += 1
    
    if better_count >= 4:
        better_counter += 1
        rows_with_3_better.append((index, x_predict.iloc[index, :], row))

# 'better'가 3개 이상인 행이 있는지 확인
if better_counter > 0:
    print(f'{better_counter}개가 더 나은 결과')
    print("더 나은 결과들의 데이터:")
    for row_num, data, prediction in rows_with_3_better:
        print(f"Row {row_num}: Data - {data.values}, Predicted values - {prediction}")
else:
    print('No rows with 3 or more "better"')




