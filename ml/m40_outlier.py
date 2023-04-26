import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
model = RandomForestRegressor()
le = LabelEncoder()
imputer = IterativeImputer(XGBRegressor())
scaler = MinMaxScaler()

def split_xy(data):
    data = pd.DataFrame(imputer.fit_transform(data))
    data_x, data_y = np.array(data.drop(data.shape[1]-1, axis=1)), data[data.shape[1]-1]
    return data_x, data_y
def outliers(a):
    b = []
    for i in range(a.shape[1]):
        q1, q3 = np.percentile(a[:, i], [25, 75], axis=0)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - (iqr * 1.5), q3 + (iqr * 1.5)
        b.append(np.where((a[:, i]>upper_bound)|(a[:, i]<lower_bound))[0])
    return b
def Runmodel(a, x, y):
    x = imputer.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(a , 'result : ', result)

path_d_ddarung = './_data/ddarung/'
path_k_bike = './_data/kaggle_bike/'
path_d_wine = './_data/wine/'
path_d_diabetes = './_data/dacon_diabetes/'

ddarung = pd.read_csv(path_d_ddarung + 'train.csv', index_col=0)
ddarung_x, ddarung_y = split_xy(ddarung)

bike = pd.read_csv(path_k_bike + 'train.csv', index_col=0)
bike_x, bike_y = split_xy(bike)

diabetes = pd.read_csv(path_d_diabetes + 'train.csv', index_col=0)
diabetes_x, diabetes_y = split_xy(diabetes)

wine = pd.read_csv(path_d_wine + 'train.csv', index_col=0)
wine['type'] = le.fit_transform(wine['type'])
wine = pd.DataFrame(imputer.fit_transform(wine))
wine_x = np.array(wine.drop(0, axis=1))
wine_y = wine[0]

out_ddarung = outliers(ddarung_x)
out_bike = outliers(bike_x)
out_wine = outliers(wine_x)
out_diabetes = outliers(diabetes_x)

var_list = [ddarung_x, bike_x, wine_x, diabetes_x, ddarung_y, bike_y, wine_y, diabetes_y]
out_list = [out_ddarung, out_bike, out_wine, out_diabetes]
name_list = ['ddarung', 'bike', 'wine', 'diabetes']

for j in range(len(out_list)):
    for i in range(var_list[j].shape[1]):
        var_list[j][out_list[j][i], i] = None
    if j >= 2:
        model = RandomForestClassifier()
    Runmodel(name_list[j], var_list[j], var_list[j+4])