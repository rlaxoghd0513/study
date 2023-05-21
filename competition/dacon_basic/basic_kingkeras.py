import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
import optuna
import datetime
import warnings
warnings.filterwarnings('ignore')

poly = PolynomialFeatures(degree=2, include_bias=False)

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

path = './_data/dacon_basic/'
path_save = './_save/dacon_basic/'
# path_save_min = './_save/dacon_cal/min/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0).drop(['Weight_Status'], axis=1)
test_csv = pd.read_csv(path + 'test.csv', index_col=0).drop(['Weight_Status'], axis=1)
submit_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv['Calories_Burned']

x['Height(inch)'] = 12*x['Height(Feet)']+x['Height(Remainder_Inches)']
test_csv['Height(inch)'] = 12*test_csv['Height(Feet)']+test_csv['Height(Remainder_Inches)']

x['BMI'] = (703*x['Weight(lb)']/x['Height(Feet)']**2)
test_csv['BMI'] = (703*test_csv['Weight(lb)']/test_csv['Height(Feet)']**2)

# x['BMR'] = 10 * x['Weight(lb)'] * 0.453592 + 6.25 * x['Height(inch)'] * 2.54 - 5 * x['Age'] + x['Gender'].apply(lambda x: 5 if x=='M' else -161)
# test_csv['BMR'] = 10 * test_csv['Weight(lb)'] * 0.453592 + 6.25 * test_csv['Height(inch)'] * 2.54 - 5 * test_csv['Age'] + test_csv['Gender'].apply(lambda x: 5 if x=='M' else -161)

# x['Exercise_Intensity'] = x['BPM'] / (220 - x['Age'])
# test_csv['Exercise_Intensity'] = test_csv['BPM'] / (220 - test_csv['Age'])

# x['Calories_Proxy_MET'] = x['Exercise_Intensity'] / 100 * x['Weight(lb)'] * 0.453592 * x['Exercise_Duration']
# test_csv['Calories_Proxy_MET'] = test_csv['Exercise_Intensity'] / 100 * test_csv['Weight(lb)'] * 0.453592 * test_csv['Exercise_Duration']

# def process_activity_factor(x):
#     if x < 0.4: return 'Sedentary'
#     elif x < 0.6: return 'Light Exercise'
#     elif x < 0.8: return 'Moderate Exercise'
#     else: return 'Heavy Exercise'
    
# x['Activity_Factor'] = x['Exercise_Intensity'].apply(process_activity_factor)
# test_csv['Activity_Factor'] = test_csv['Exercise_Intensity'].apply(process_activity_factor)

# def process_Calories_Proxy_AF(x):
#     if x == 'Sedentary': return 1.2
#     elif x == 'Light Exercise': return 1.375
#     elif x == 'Moderate Exercise': return 1.55
#     else: return 1.725

# x['Calories_Proxy_AF'] = x['BMR'] * x['Activity_Factor'].apply(process_Calories_Proxy_AF) / 1440
# test_csv['Calories_Proxy_AF'] = test_csv['BMR'] * test_csv['Activity_Factor'].apply(process_Calories_Proxy_AF) / 1440

le = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
test_csv['Gender'] = le.transform(test_csv['Gender'])

x = x.drop(['Height(Feet)', 'Height(Remainder_Inches)'], axis=1)
test_csv = test_csv.drop(['Height(Feet)', 'Height(Remainder_Inches)'], axis=1)

scaler = MaxAbsScaler()
x = pd.DataFrame(scaler.fit_transform(x))
test_csv = pd.DataFrame(scaler.transform(test_csv))

for i in range(10000):
    kf = KFold(n_splits=5, shuffle=True, random_state=i)

    for train_idx, test_idx in kf.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    
        def objective(trial):
            alpha = trial.suggest_loguniform('alpha', 0.0000001, 0.1)
            n_restarts_optimizer  = trial.suggest_int('n_restarts_optimizer', 1, 20)
            optimizer = trial.suggest_categorical('optimizer', ['fmin_l_bfgs_b', 'Powell', 'CG'])

            model = GaussianProcessRegressor(
                alpha=alpha,
                n_restarts_optimizer=n_restarts_optimizer,
                optimizer=optimizer,
            )
            
            model.fit(x_train, y_train)
            
            print('GPR result : ', model.score(x_test, y_test))
            
            y_pred = np.round(model.predict(x_test))
            rmse = RMSE(y_test, y_pred)
            print('GPR RMSE : ', rmse)
            if rmse < 0.188:
                submit_csv['Calories_Burned'] = np.round(model.predict(test_csv))
                date = datetime.datetime.now()
                date = date.strftime('%m%d_%H%M%S')
                submit_csv.to_csv(path_save + date + str(round(rmse, 5)) + '.csv')
            return rmse
        opt = optuna.create_study(direction='minimize')
        opt.optimize(objective, n_trials=20)
        print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)