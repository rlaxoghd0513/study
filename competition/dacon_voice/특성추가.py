import random
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
import librosa

from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings(action='ignore')

CFG = {
    'SR': 80000,  # 높으면 잘 잘라줌
    'N_MFCC': 128,  # Melspectrogram 벡터를 추출할 개수
    'SEED': 42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED'])  # Seed 고정

train_df = pd.read_csv('./_data/dacon_voice/train.csv')
test_df = pd.read_csv('./_data/dacon_voice/test.csv')

train_df['path'] = './_data/dacon_voice/' + train_df['path']
test_df['path'] = './_data/dacon_voice/' + test_df['path']

scaler = MinMaxScaler()

def get_mfcc_feature(df):
    features = []
    for path in tqdm(df['path']):
        y, sr = librosa.load(path, sr=CFG['SR'])
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG['N_MFCC'])
        features.append({
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_max': np.max(mfcc, axis=1),
            'mfcc_min': np.min(mfcc, axis=1),
            'mfcc_var': np.var(mfcc, axis=1),
            'mfcc_median': np.median(mfcc, axis=1),
        })

    mfcc_df = pd.DataFrame(features)
    mfcc_mean_df = pd.DataFrame(mfcc_df['mfcc_mean'].tolist(), columns=[f'mfcc_mean_{i}' for i in range(CFG['N_MFCC'])])
    mfcc_max_df = pd.DataFrame(mfcc_df['mfcc_max'].tolist(), columns=[f'mfcc_max_{i}' for i in range(CFG['N_MFCC'])])
    mfcc_min_df = pd.DataFrame(mfcc_df['mfcc_min'].tolist(), columns=[f'mfcc_min_{i}' for i in range(CFG['N_MFCC'])])
    mfcc_var_df = pd.DataFrame(mfcc_df['mfcc_var'].tolist(), columns=[f'mfcc_var_{i}' for i in range(CFG['N_MFCC'])])
    mfcc_median_df = pd.DataFrame(mfcc_df['mfcc_median'].tolist(), columns=[f'mfcc_median_{i}' for i in range(CFG['N_MFCC'])])

    mfcc_mean_normalized = pd.DataFrame(scaler.fit_transform(mfcc_mean_df), columns=[f'mfcc_mean_norm_{i}' for i in range(CFG['N_MFCC'])])  # 정규화 추가
    mfcc_max_normalized = pd.DataFrame(scaler.fit_transform(mfcc_max_df), columns=[f'mfcc_max_norm_{i}' for i in range(CFG['N_MFCC'])])  # 정규화 추가
    mfcc_min_normalized = pd.DataFrame(scaler.fit_transform(mfcc_min_df), columns=[f'mfcc_min_norm_{i}' for i in range(CFG['N_MFCC'])])  # 정규화 추가
    mfcc_var_normalized = pd.DataFrame(scaler.fit_transform(mfcc_var_df), columns=[f'mfcc_var_norm_{i}' for i in range(CFG['N_MFCC'])])  # 정규화 추가
    mfcc_median_normalized = pd.DataFrame(scaler.fit_transform(mfcc_median_df), columns=[f'mfcc_median_norm_{i}' for i in range(CFG['N_MFCC'])])  # 정규화 추가

    mfcc_df = pd.concat([mfcc_mean_normalized, mfcc_max_normalized, mfcc_min_normalized, mfcc_var_normalized, mfcc_median_normalized], axis=1)

    return mfcc_df


def get_feature_contrast(df):
    features = []
    for path in tqdm(df['path']):
        y, sr = librosa.load(path, sr=CFG['SR'])
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=2048, hop_length=1024)
        features.append({
            'contrast_mean': np.mean(contrast, axis=1),
            'contrast_max': np.max(contrast, axis=1),
            'contrast_min': np.min(contrast, axis=1),
            'contrast_var': np.var(contrast, axis=1),
            'contrast_median': np.median(contrast, axis=1),
        })

    contrast_df = pd.DataFrame(features)
    contrast_mean_df = pd.DataFrame(contrast_df['contrast_mean'].tolist(), columns=[f'contrast_mean_{i}' for i in range(contrast.shape[0])])
    contrast_max_df = pd.DataFrame(contrast_df['contrast_max'].tolist(), columns=[f'contrast_max_{i}' for i in range(contrast.shape[0])])
    contrast_min_df = pd.DataFrame(contrast_df['contrast_min'].tolist(), columns=[f'contrast_min_{i}' for i in range(contrast.shape[0])])
    contrast_var_df = pd.DataFrame(contrast_df['contrast_var'].tolist(), columns=[f'contrast_var_{i}' for i in range(contrast.shape[0])])
    contrast_median_df = pd.DataFrame(contrast_df['contrast_median'].tolist(), columns=[f'contrast_median_{i}' for i in range(contrast.shape[0])])

    contrast_df = pd.concat([contrast_mean_df, contrast_max_df, contrast_min_df, contrast_var_df, contrast_median_df], axis=1)

    return contrast_df


train_mf = get_mfcc_feature(train_df)
test_mf = get_mfcc_feature(test_df)

train_contrast = get_feature_contrast(train_df)
test_contrast = get_feature_contrast(test_df)

train_x = pd.concat([train_mf, train_contrast], axis=1)
test_x = pd.concat([test_mf, test_contrast], axis=1)

train_y = train_df['label']

train_x['label'] = train_df['label']

from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=CFG['SEED'])

train_data = TabularDataset(train_x)
val_data = TabularDataset(val_x)
test_data = TabularDataset(test_x)

label = 'label'
eval_metric = 'accuracy'
time_limit = 3600 * 1

predictor = TabularPredictor(
    label=label, eval_metric=eval_metric
).fit(train_data, presets='best_quality', time_limit=time_limit, ag_args_fit={'num_gpus': 0, 'num_cpus': 4})

model_to_use = predictor.get_model_best()
model_score = predictor.evaluate(val_data, model=model_to_use)
model_pred = predictor.predict(test_data, model=model_to_use)

print("Best Model Score:", model_score)

accuracy = model_score['accuracy']
accuracy_rounded = round(accuracy, 4)

submission = pd.read_csv('./_data/dacon_voice/sample_submission.csv')
submission['label'] = model_pred

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
save_path = './_save/dacon_voice/'

submission.to_csv(save_path + date + '_' + str(accuracy_rounded) + '.csv', index= False)