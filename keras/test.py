import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#데이터
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=4)

print(train_csv) #(10886,11)
print(train_csv.shape)
print(train_csv.info())
#   datetime    10886 non-null  object
#  1   holiday     10886 non-null  int64
#  2   workingday  10886 non-null  int64
#  3   weather     10886 non-null  int64
#  4   temp        10886 non-null  float64
#  5   atemp       10886 non-null  float64
#  6   humidity    10886 non-null  int64
#  7   windspeed   10886 non-null  float64
#  8   casual      10886 non-null  int64
#  9   registered  10886 non-null  int64
#  10  count 
print(train_csv.columns)
#(['datetime', 'season', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object')
 
test_csv = pd.read_csv(path + 'test.csv', index_col=1)
print(test_csv)
print(test_csv.shape)  #(6493, 8)
print(test_csv.info())
print(test_csv.columns)