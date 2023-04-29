import pandas as pd
import os
import numpy as np
from typing import List,Tuple


def file_path()->str:
    return './_data/ai_spark/'


def sort_data_by_first_element(*args: List[List]) -> Tuple[List]:
    return tuple(map(list, zip(*sorted(zip(*args), key=lambda x: x[0]))))


def create_awsmap_pmmap() -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = file_path()
    path_list = os.listdir(path)
    meta = '/'.join([path, path_list[1]])
    meta_list = os.listdir(meta)
    file_name_aws = meta_list[0]
    awsmap = pd.read_csv('/'.join([meta, file_name_aws]))
    awsmap = awsmap.drop(awsmap.columns[-1], axis=1)
    file_name_pm = meta_list[1]
    pmmap = pd.read_csv('/'.join([meta, file_name_pm]))
    pmmap = pmmap.drop(pmmap.columns[-1], axis=1)
    return awsmap, pmmap


def create_train(filename:str)->pd.DataFrame:
    path=file_path()
    path_list=os.listdir(path)    
    train='/'.join([path,path_list[4]])
    return pd.read_csv(f'{train}/{filename}.csv')


def create_train_AWS(filename:str)->pd.DataFrame:
    path=file_path()
    path_list=os.listdir(path)  
    train_aws='/'.join([path,path_list[5]])
    return pd.read_csv(f'{train_aws}/{filename}.csv')


def create_test_input(filename:str)->pd.DataFrame:
    path=file_path()
    path_list=os.listdir(path)    
    test_input='/'.join([path,path_list[3]])
    return pd.read_csv(f'{test_input}/{filename}.csv')

def create_test_AWS(filename:str)->pd.DataFrame:
    path=file_path()
    path_list=os.listdir(path)  
    test_aws='/'.join([path,path_list[2]])
    return pd.read_csv(f'{test_aws}/{filename}.csv')

############################################거리자료############################################
def all_distance_info() -> pd.DataFrame:
    awsmap, pmmap = create_awsmap_pmmap()
    aws_name=[]
    pm_name=[]
    Distance=[]
    for i in range(awsmap.shape[0]):
        for j in range(pmmap.shape[0]):
            aws_name.append(awsmap["Location"][i])
            pm_name.append(pmmap["Location"][j])
            LatitudeDis=awsmap["Latitude"][i]-pmmap["Latitude"][j]
            LongitudeDis=awsmap["Longitude"][i]-pmmap["Longitude"][j]
            Distance.append(np.sqrt(LatitudeDis**2+LongitudeDis**2))
    pm_name,aws_name,Distance=sort_data_by_first_element(pm_name,aws_name,Distance)
    aws_name=pd.Series(aws_name,name='aws_name')
    pm_name=pd.Series(pm_name,name='pm_name')
    Distance=pd.Series(Distance,name='Distance')
    LocationInfo=pd.concat([aws_name,pm_name,Distance],axis=1)
    return LocationInfo

##########################################최인접 자료#########################################

def min_distance_info()->pd.DataFrame:
    awsmap, pmmap = create_awsmap_pmmap()
    aws_name=[]
    pm_name=[]
    Distance=[]
    for j in range(pmmap.shape[0]):
        min_distance=np.sqrt(2)*180
        min_awd=''
        pm_name.append(pmmap["Location"][j])
        for i in range(awsmap.shape[0]):
            LatitudeDis=awsmap["Latitude"][i]-pmmap["Latitude"][j]
            LongitudeDis=awsmap["Longitude"][i]-pmmap["Longitude"][j]
            current_dis=np.sqrt(LatitudeDis**2+LongitudeDis**2)
            if current_dis<min_distance:
                min_awd=awsmap["Location"][i]
                min_distance=current_dis
        aws_name.append(min_awd)
        Distance.append(min_distance)
    pm_name,aws_name,Distance=sort_data_by_first_element(pm_name,aws_name,Distance)
    aws_name=pd.Series(aws_name,name='aws_name')
    pm_name=pd.Series(pm_name,name='pm_name')
    Distance=pd.Series(Distance,name='Distance')
    LocationInfo=pd.concat([pm_name,aws_name,Distance],axis=1)
    LocationInfo
    return LocationInfo



def load_min_distance(print_download:bool=True,load_name:bool=True
                      )->Tuple[List[pd.DataFrame],List[pd.DataFrame],List[str]]:
    '''각 PM측정소의 최근접 AWS측정소 데이터와 PM측정소 데이터를 concate해서 출력해줌
    
    output: train_datas,test_datas,pm_name
    '''
    distance_info=min_distance_info()
    aws_name=distance_info['aws_name']
    pm_name=distance_info['pm_name']
    Distance=distance_info['Distance']
    train_datas=[]
    test_datas=[]
    for i in range(len(pm_name)):
        pm_train=create_train(pm_name[i])
        pm_test=create_test_input(pm_name[i])
        aws_train=create_train_AWS(aws_name[i])
        aws_test=create_test_AWS(aws_name[i])
        if print_download:
            print(f'{pm_name[i]}파일 로드 완료')
        train_datas.append(pd.concat([aws_train.drop(['지점'],axis=1),pm_train['PM2.5']],axis=1))
        test_datas.append(pd.concat([aws_test.drop(['지점'],axis=1),pm_test['PM2.5']],axis=1))
    if load_name:
        return train_datas,test_datas,pm_name
    else:
        return train_datas,test_datas

def min_distance_info_sample()->pd.DataFrame:
    awsmap, pmmap = create_awsmap_pmmap()
    aws_name=[]
    pm_name=[]
    Distance=[]
    min_distance=np.sqrt(2)*180
    min_awd=''
    for j in range(2):
        pm_name.append(pmmap["Location"][j])
        for i in range(awsmap.shape[0]):
            LatitudeDis=awsmap["Latitude"][i]-pmmap["Latitude"][j]
            LongitudeDis=awsmap["Longitude"][i]-pmmap["Longitude"][j]
            current_dis=np.sqrt(LatitudeDis**2+LongitudeDis**2)
            if current_dis<min_distance:
                min_awd=awsmap["Location"][i]
                min_distance=current_dis
        aws_name.append(min_awd)
        Distance.append(min_distance)
    pm_name,aws_name,Distance=sort_data_by_first_element(pm_name,aws_name,Distance)
    aws_name=pd.Series(aws_name,name='aws_name')
    pm_name=pd.Series(pm_name,name='pm_name')
    Distance=pd.Series(Distance,name='Distance')
    LocationInfo=pd.concat([pm_name,aws_name,Distance],axis=1)
    LocationInfo
    return LocationInfo


def load_min_distance_sample(print_download:bool=True,load_name:bool=True
                      )->Tuple[List[pd.DataFrame],List[pd.DataFrame],List[str]]:
    '''각 PM측정소의 최근접 AWS측정소 데이터와 PM측정소 데이터를 concate해서 출력해줌'''
    distance_info=min_distance_info_sample()
    aws_name=distance_info['aws_name']
    pm_name=distance_info['pm_name']
    Distance=distance_info['Distance']
    train_datas=[]
    test_datas=[]
    for i in range(len(pm_name)):
        pm_train=create_train(pm_name[i])
        pm_test=create_test_input(pm_name[i])
        aws_train=create_train_AWS(aws_name[i])
        aws_test=create_test_AWS(aws_name[i])
        if print_download:
            print(f'{pm_name[i]}파일 로드 완료')
        train_datas.append(pd.concat([aws_train.drop(['지점'],axis=1),pm_train['PM2.5']],axis=1))
        test_datas.append(pd.concat([aws_test.drop(['지점'],axis=1),pm_test['PM2.5']],axis=1))
    if load_name:
        return train_datas,test_datas,pm_name
    else:
        return train_datas,test_datas