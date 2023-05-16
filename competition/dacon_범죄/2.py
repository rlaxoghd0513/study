import pandas as pd
from unidecode import unidecode

path = './_data/dacon_범죄/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
# print(train_csv.shape) #(84406, 19)
# print(train_csv.columns) #Index(['월', '요일', '시간', '소관경찰서', '소관지역', '사건발생거리', '강수량(mm)', '강설량(mm)','적설량(cm)', 
                                # '풍향', '안개', '짙은안개', '번개', '진눈깨비', '서리', '연기/연무', '눈날림','범죄발생지', 'TARGET'],

new_columns = [unidecode(col) for col in train_csv.columns]
train_csv.columns = new_columns

new_columns1 = [unidecode(col) for col in test_csv.columns]
test_csv.columns = new_columns1

# print(train_csv.columns)
#요일 범죄발생지
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['yoil'] = le.fit_transform(train_csv['yoil'])
test_csv['yoil'] = le.transform(test_csv['yoil'])

train_csv['beomjoebalsaengji'] = le.fit_transform(train_csv['beomjoebalsaengji'])
test_csv['beomjoebalsaengji'] = le.transform(test_csv['beomjoebalsaengji'])

print(le.classes_)
# ['공원' '백화점' '병원' '식당' '약국' '은행' '인도' '주거지' '주유소' '주차장' '차도' '편의점' '학교','호텔/모텔']
# 공원         736      0
# 학교         728      12
# 약국         653       4
# 호텔/모텔      591      13
# 병원         453       2
# 은행         132      5

# print(train_csv.isnull().sum())
# print(train_csv.info())
# 빈도 낮은 범죄지역 제거
# x = x[x['beomjoebalsaengji'] != value]
train_csv = train_csv[~train_csv['beomjoebalsaengji'].isin([0,2,4,5,12,13])]

########################################이상치##########################################
list = ['sageonbalsaenggeori','gangsuryang(mm)','gangseolryang(mm)','jeogseolryang(cm)']
def remove_outliers_iqr(df, cols, factor=1.5):
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

train_csv = remove_outliers_iqr(train_csv, list, factor=1.5)
#####################################################################################
x = train_csv.drop('TARGET', axis=1)
y = train_csv['TARGET']


# print(x.shape, y.shape)#(84406, 18) (84406,)
############################################################################
# print(x.corr()) #Correlation 상관    

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=1.2)
# sns.heatmap(data=x.corr(), square=True, annot=True, cbar=True)
# plt.show()

#소관뭐시기 두개 합하기
x['new'] = x['sogwangyeongcalseo']+x['sogwanjiyeog']
test_csv['new'] = test_csv['sogwangyeongcalseo']+test_csv['sogwanjiyeog']
x = x.drop('sogwangyeongcalseo', axis=1)
x = x.drop('sogwanjiyeog', axis=1)
test_csv = test_csv.drop('sogwangyeongcalseo', axis=1)
test_csv = test_csv.drop('sogwanjiyeog', axis=1)

#진눈깨비 서리 합하기
x['new2'] = x['jinnunggaebi']+x['seori']
test_csv['new2'] = test_csv['jinnunggaebi']+test_csv['seori']
x = x.drop('jinnunggaebi', axis=1)
test_csv = test_csv.drop('jinnunggaebi', axis=1)
x = x.drop('seori', axis=1)
test_csv = test_csv.drop('seori', axis=1)

print(x.columns)
#사건발생거리 강수량 적설량 
# x['sageonbalsaenggeori'] = x['sageonbalsaenggeori'].round(3)
# test_csv['sageonbalsaenggeori'] = test_csv['sageonbalsaenggeori'].round(3)
# x['gangsuryang(mm)'] = x['gangsuryang(mm)'].round(3)
# test_csv['gangsuryang(mm)'] = test_csv['gangsuryang(mm)'].round(3)
# x['jeogseolryang(cm)'] = x['jeogseolryang(cm)'].round(3)
# test_csv['jeogseolryang(cm)'] = test_csv['jeogseolryang(cm)'].round(3)


print(x.shape, test_csv.shape)#(84406, 16) (17289, 16) -> (81113, 16) (17289, 16) -> 이상치 (57008, 16) (17289, 16)
#######################################################################
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.decomposition import PCA
# pca = PCA(n_components=14)
# x = pca.fit_transform(x)
# print(x.shape) #(84406, 2)
# test_csv = pca.transform(test_csv)
# print(test_csv.shape)
##############################################################################
# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(degree=2)
# x = poly.fit_transform(x)
# test_csv = poly.transform(test_csv)
# print(x.shape, test_csv.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=42, stratify=y)
# #################################################################################################################

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler() # robust 이상치? minmax standard
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# ###################################################################################################################
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV, cross_val_score, cross_val_predict, RandomizedSearchCV

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

parameters = {
    'num_leaves': [31, 127],
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [100,200],
    'subsample': [0.5, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0],
    'reg_alpha': [0.1, 0.5, 1.0],
    'reg_lambda': [0.1, 0.5, 1.0],
    'min_child_samples': [10, 20, 30],
    'min_child_weight': [1e-4, 1e-3, 1e-2],
    'max_depth': [-1, 5, 10],
}

lgbm = LGBMClassifier()
model = RandomizedSearchCV(lgbm, parameters, cv=kfold, n_jobs=-1, refit=True, verbose=1)

model.fit(x_train, y_train)
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_feature_importances(model):                
#         n_features = x.shape[1]
#         plt.barh(np.arange(n_features),model.feature_importances_,align='center') 
                                                                            
#         plt.yticks(np.arange(n_features),x.columns) 
                                                            
#         plt.xlabel('Feature Importances')                        
#         plt.ylabel('Features')
#         plt.ylim(-1, n_features) 
        
  
#         plt.title('XGBClassifier')
    
# plt.subplot(2,2,2)
# plot_feature_importances(model)
# plt.show()
# print(x.columns)
# print(test_csv.columns)

print('최적매개변수:', model.best_estimator_)
print('model_score:', model.score(x_test, y_test))

y_predict = model.best_estimator_.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score
acc = accuracy_score(y_test, y_predict)
print('acc:', acc)

f1 = f1_score(y_test, y_predict, average= 'macro')
print('f1:', f1)

path_save = './_save/dacon_범죄/'

y_submit = model.best_estimator_.predict(test_csv)

submission = pd.read_csv(path + 'sample_submission.csv', index_col =0)

submission['TARGET'] = y_submit

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(path_save + date + '_' + str(round(f1, 2)) + '.csv')