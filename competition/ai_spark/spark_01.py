import numpy as np
import pandas as pd
import datetime
import glob

path = './_data/ai_spark/'
path_save = './_save/ai_spark/'
train_csv_list = ['공주.csv','노은동.csv','논산.csv','대천2동.csv','독곶리.csv','동문동.csv','모종동.csv','문창동.csv',]
for i in train_csv_list:
    train_data = pd.read_csv(path + i)
    