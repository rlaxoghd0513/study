import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor

path_1 = './_data/ai_spark/TRAIN_AWS/'
공주 = pd.read_csv(path_1 + '공주.csv', index_col=None)
세종금남 = pd.read_csv(path_1 + '세종금남.csv', index_col=None)
논산 = pd.read_csv(path_1 + '논산.csv', index_col=None)
대천항 = pd.read_csv(path_1 + '대천항.csv', index_col=None)
대산 = pd.read_csv(path_1 + '대산.csv', index_col = None)
당진 = pd.read_csv(path_1 + '당진.csv', index_col=None)
세종전의 = pd.read_csv(path_1 + '세종전의.csv', index_col = None)
세천 = pd.read_csv(path_1 + '세천.csv', index_col = None)
성거 = pd.read_csv(path_1 + '성거.csv', index_col = None)
세종연서 = pd.read_csv(path_1 + '세종연서.csv', index_col = None)
세종고운 = pd.read_csv(path_1 + '세종고운.csv', index_col = None)
예산 = pd.read_csv(path_1 + '예산.csv', index_col = None)
장동 = pd.read_csv(path_1 + '장동.csv', index_col = None)
태안 = pd.read_csv(path_1 + '태안.csv', index_col = None)
오월드 = pd.read_csv(path_1 + '오월드.csv', index_col = None)
홍북 = pd.read_csv(path_1 + '홍북.csv', index_col = None)

train_aws_files = pd.concat([공주,세종금남, 논산,대천항,대산,당진,세종전의,세천,성거,세종전의,세종연서,세종고운,
                   예산,장동,태안,오월드,홍북])
print(train_aws_files)
train_aws_files= train_aws_files.drop(['연도','일시','지점'],  axis=1)
print(train_aws_files)

