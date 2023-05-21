import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import optuna
import datetime
import warnings
warnings.filterwarnings('ignore')

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

path = './_data/dacon_basic/'
save_path = './_save/dacon_basic/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0).drop(['Weight_Status'], axis=1)
test_csv = pd.read_csv(path + 'test.csv', index_col=0).drop(['Weight_Status'], axis=1)
submit_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv['Calories_Burned']

x['Height(Feet)'] = 12*x['Height(Feet)']+x['Height(Remainder_Inches)']
x['Height(Remainder_Inches)'] = 703*x['Weight(lb)']/x['Height(Feet)']**2

test_csv['Height(Feet)'] = 12*test_csv['Height(Feet)']+test_csv['Height(Remainder_Inches)']
test_csv['Height(Remainder_Inches)'] = 703*test_csv['Weight(lb)']/test_csv['Height(Feet)']**2

le = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
test_csv['Gender'] = le.transform(test_csv['Gender'])

scaler_list = [MinMaxScaler(), MaxAbsScaler(), StandardScaler(), RobustScaler()]
model_list = [CatBoostRegressor()]


def objective(trial, x_train, y_train, x_test, y_test, scaler, model):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    depth = trial.suggest_int('depth', 4, 10)
    l2_leaf_reg = trial.suggest_loguniform('l2_leaf_reg', 1e-4, 1)

    model = CatBoostRegressor(
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
    )

    model.fit(x_train, y_train, eval_set=(x_test, y_test), verbose=False)
    y_pred = model.predict(x_test)
    rmse = RMSE(y_test, y_pred)

    return rmse

min_rmse = 1
for k in range(1000000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=k)

    for scaler in scaler_list:
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        test_csv_scaled = scaler.transform(test_csv)

        for model in model_list:
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, x_train_scaled, y_train, x_test_scaled, y_test, scaler, model), n_trials=50)

            best_params = study.best_params
            best_rmse = study.best_value

            if best_rmse < min_rmse:
                min_rmse = best_rmse
                submit_csv['Calories_Burned'] = model.predict(test_csv_scaled)
                date = datetime.datetime.now()
                date = date.strftime('%m%d_%H%M%S')
                submit_csv.to_csv(save_path + date + str(round(best_rmse, 5)) + '.csv')
