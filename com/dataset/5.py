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
# 시각화

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path ='C:\\Windows\\Fonts\\gulim.ttc'
fontprop = fm.FontProperties(fname=font_path, size=14)

# X1부터 X30까지의 칼럼 선택
selected_columns = train_csv.iloc[:, :30]
correlation_matrix = selected_columns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('x1에서 x30까지의 상관관계',fontproperties=fontprop)
plt.show()

#상관관계가 0.99넘는 열들 출력
import numpy as np

correlation_matrix = train_csv.iloc[:, :30].corr()

high_corr = np.where((correlation_matrix.abs() >= 0.99) & (correlation_matrix.abs() < 1))

print("0.99이상인 상관관계")
for i, j in zip(*high_corr):
    if i != j and i < j:
        print(f"'{train_csv.columns[i]}','{train_csv.columns[j]}' ,{correlation_matrix.iloc[i, j]:.2f}")

# 'X12','X13' ,0.99
# 'X12','X30' ,-0.99
# 'X13','X22' ,1.00
# 'X13','X25' ,1.00
# 'X13','X30' ,-0.99
# 'X22','X25' ,1.00


# X1~X30이랑 A, B, C, D, E 칼럼 사이 상관관계
selected_columns = train_csv.iloc[:, :30].join(train_csv.iloc[:, -5:])
correlation_matrix = selected_columns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix.iloc[:30, -5:], annot=True, cmap='coolwarm', fmt=".2f")
plt.title('x1~30과 ABCDE의 상관관계',fontproperties=fontprop)
plt.show()

# A, B, C, D, E 칼럼 서로의 상관관계 
selected_columns = train_csv[['A', 'B', 'C', 'D', 'E']]
correlation_matrix = selected_columns.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('ABCDE끼리의 상관관계',fontproperties=fontprop)
plt.show()


# # # 각 열의 분포 시각화 
# for column in train_csv.columns:
#     sns.histplot(train_csv[column], kde=True) 
#     plt.title(f'{column}의 분포',fontproperties=fontprop)
#     plt.xlabel('Values')
#     plt.ylabel('Frequency')
#     plt.show()
    
##################################################################################################################
#0.99넘는 높은 상관관계 열끼리 feature engineering

train_csv.insert(loc=0, column='col_6', value=train_csv.iloc[:, 21] + train_csv.iloc[:, 24])
train_csv.insert(loc=0, column='col_5', value=train_csv.iloc[:, 12] + train_csv.iloc[:, 29])
train_csv.insert(loc=0, column='col_4', value=train_csv.iloc[:, 12] + train_csv.iloc[:, 24])
train_csv.insert(loc=0, column='col_3', value=train_csv.iloc[:, 12] + train_csv.iloc[:, 21])
train_csv.insert(loc=0, column='col_2', value=train_csv.iloc[:, 11] + train_csv.iloc[:, 29])
train_csv.insert(loc=0, column='col_1', value=train_csv.iloc[:, 11] + train_csv.iloc[:, 12])

# print(train_csv)#[43865 rows x 41 columns]

test_csv.insert(loc=0, column='col_6', value=test_csv.iloc[:, 21] + test_csv.iloc[:, 24])
test_csv.insert(loc=0, column='col_5', value=test_csv.iloc[:, 12] + test_csv.iloc[:, 29])
test_csv.insert(loc=0, column='col_4', value=test_csv.iloc[:, 12] + test_csv.iloc[:, 24])
test_csv.insert(loc=0, column='col_3', value=test_csv.iloc[:, 12] + test_csv.iloc[:, 21])
test_csv.insert(loc=0, column='col_2', value=test_csv.iloc[:, 11] + test_csv.iloc[:, 29])
test_csv.insert(loc=0, column='col_1', value=test_csv.iloc[:, 11] + test_csv.iloc[:, 12])

# print(test_csv)#[18798 rows x 41 columns]

##################################################################################################################
# 낮은 상관관계 제거할 열
cols_to_drop = ['X1','X5', 'X10', 'X14']

train_csv = train_csv.drop(cols_to_drop, axis=1)
test_csv = test_csv.drop(cols_to_drop, axis=1)

##################################################################################################################

x = train_csv.iloc[:, :32]  
y = train_csv.iloc[:, 32:] 
# print(x.shape, y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=1336, shuffle=True)

#####################################################################################################################
#분포에 따라 따로따로 스케일링
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

x_predict = test_csv.iloc[:, :32]  
y_predict = test_csv.iloc[:, 32:]

# print(x_train.shape)#(35092, 32)
# print(x_test.shape)#(8773, 32)
# print(x_predict.shape)#(18798, 32)
    
##################################################################################################################################
# #polynomial feature

# from sklearn.preprocessing import PolynomialFeatures

# poly = PolynomialFeatures(degree=2) 

# x_train_poly = poly.fit_transform(x_train)
# x_test_poly = poly.transform(x_test)
# x_predict_poly = poly.transform(x_predict)

# x_train = pd.concat([x_train.reset_index(drop=True), pd.DataFrame(x_train_poly[:, len(x_train.columns):])], axis=1)
# x_test = pd.concat([x_test.reset_index(drop=True), pd.DataFrame(x_test_poly[:, len(x_test.columns):])], axis=1)
# x_predict = pd.concat([x_predict.reset_index(drop=True), pd.DataFrame(x_predict_poly[:, len(x_predict.columns):])], axis=1)

# print(x_train.shape)
# print(x_test.shape)
# print(x_predict.shape)

########################################################################################################################################
#pca
# from sklearn.decomposition import PCA

# n_components = 15 

# pca = PCA(n_components=n_components)

# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)
# x_predict = pca.transform(x_predict)

# print(x_train.shape)
# print(x_test.shape)
# print(x_predict.shape)

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


model_A = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=param_grid_rf, n_iter=100, cv=3, verbose=2, random_state=1336, n_jobs=-1)
model_B = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=param_grid_rf, n_iter=100, cv=3, verbose=2, random_state=1336, n_jobs=-1)
model_C = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=param_grid_rf, n_iter=100, cv=3, verbose=2, random_state=1336, n_jobs=-1)
model_D = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=param_grid_rf, n_iter=100, cv=3, verbose=2, random_state=1336, n_jobs=-1)
model_E = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=param_grid_rf, n_iter=100, cv=3, verbose=2, random_state=1336, n_jobs=-1)


multi_output_model = MultiOutputRegressor(estimator=RandomForestRegressor())

multi_output_model.estimators_ = [model_A, model_B, model_C, model_D, model_E]

multi_output_model.fit(x_train, y_train)

predictions = multi_output_model.predict(x_test)

mae = mean_absolute_error(y_test, predictions)
print('mae:', mae)

predict_test_csv = multi_output_model.predict(x_predict)
mae_predict = mean_absolute_error(y_predict, predict_test_csv)
print('mae_test_csv:', mae_predict)

###############################################################################################
from collections import Counter

most_common_values = {
    'A': Counter(train_csv['A']).most_common(1)[0][0],
    'B': Counter(train_csv['B']).most_common(1)[0][0],
    'C': Counter(train_csv['C']).most_common(1)[0][0],
    'D': Counter(train_csv['D']).most_common(1)[0][0],
    'E': Counter(train_csv['E']).most_common(1)[0][0]
}

predicted_values = multi_output_model.predict(x_predict)

better_counter = 0  
rows_with_3_better = [] 

for index, row in enumerate(predicted_values):
    better_count = 0 
    for i, value in enumerate(row):
        if i < 3:
            if value > most_common_values[chr(65 + i)]:
                better_count += 1
        else:
            if value < most_common_values[chr(65 + i)]:
                better_count += 1
    
    if better_count >= 3:
        better_counter += 1
        rows_with_3_better.append((index, row))


if better_counter > 0:
    print(f'better가 3개 이상인 데이터 갯수 : {better_counter}')
    for idx, prediction in rows_with_3_better:
        print(f"행 {idx}: 예측값: {prediction}")
else:
    print('결과없음')

