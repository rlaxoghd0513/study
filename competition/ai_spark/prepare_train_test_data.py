import os
from read_file import load_min_distance
from preprocessing import split_month_day_hour,Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
import pandas as pd
from typing import Tuple,List
def save_concated_data_min_distance(print_download:bool=True)->None:
    train_datas,test_datas,pm_name=load_min_distance(print_download=print_download,load_name=True)
    for i in range(len(train_datas)):
        train_datas[i]=split_month_day_hour(train_datas[i])
        test_datas[i]=split_month_day_hour(test_datas[i])
        train_datas[i]=Imputation(train_datas[i])
        os.makedirs('./_data/ai_spark/', exist_ok=True)
        os.makedirs('./_data/ai_spark/for_Train', exist_ok=True)
        os.makedirs('./_data/ai_spark/for_Train/test/', exist_ok=True)
        os.makedirs('./_data/ai_spark/for_Train/train/', exist_ok=True)  
        train_datas[i].to_csv(f'./_data/ai_spark/for_Train/train/{pm_name[i]}.csv',index=False)
        test_datas[i].to_csv(f'./_data/ai_spark/for_Train/test/{pm_name[i]}.csv',index=False)
        if print_download:
            print(f'{pm_name[i]}저장 완료!')

def load_concated_data_min_distance(print_download:bool=True)->Tuple[List[pd.DataFrame],List[pd.DataFrame],List[str]]:
    train_datas,test_datas,pm_name=load_min_distance(print_download=print_download,load_name=True)
    for i in range(len(train_datas)):
        train_datas[i]=pd.read_csv(f'./_data/ai_spark/for_Train/train/{pm_name[i]}.csv')
        test_datas[i]=pd.read_csv(f'./_data/ai_spark/for_Train/test/{pm_name[i]}.csv')
    if print_download:
        print(f'{pm_name[i]}로드 완료!')
    return train_datas,test_datas,pm_name