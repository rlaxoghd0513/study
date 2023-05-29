import random
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
import librosa

from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler, PowerTransformer, QuantileTransformer

import warnings
warnings.filterwarnings(action='ignore')

CFG = {
    'SR':22000, #높으면 잘 잘라줌
    'N_MFCC':128, # Melspectrogram 벡터를 추출할 개수
    'SEED':42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED']) # Seed 고정


train_df = pd.read_csv('./_data/train.csv')
test_df = pd.read_csv('./_data/test.csv')

train_df['path'] = './_data/' + train_df['path']
test_df['path'] = './_data/' + test_df['path']

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
        })

    mfcc_df = pd.DataFrame(features)
    mfcc_mean_df = pd.DataFrame(mfcc_df['mfcc_mean'].tolist(), columns=[f'mfcc_mean_{i}' for i in range(CFG['N_MFCC'])])
    mfcc_max_df = pd.DataFrame(mfcc_df['mfcc_max'].tolist(), columns=[f'mfcc_max_{i}' for i in range(CFG['N_MFCC'])])
    mfcc_min_df = pd.DataFrame(mfcc_df['mfcc_min'].tolist(), columns=[f'mfcc_min_{i}' for i in range(CFG['N_MFCC'])])

    mfcc_mean_normalized = pd.DataFrame(scaler.fit_transform(mfcc_mean_df), columns=[f'mfcc_mean_norm_{i}' for i in range(CFG['N_MFCC'])])  # 정규화 추가
    mfcc_max_normalized = pd.DataFrame(scaler.fit_transform(mfcc_max_df), columns=[f'mfcc_max_norm_{i}' for i in range(CFG['N_MFCC'])])  # 정규화 추가
    mfcc_min_normalized = pd.DataFrame(scaler.fit_transform(mfcc_min_df), columns=[f'mfcc_min_norm_{i}' for i in range(CFG['N_MFCC'])])  # 정규화 추가

    mfcc_df = pd.concat([mfcc_mean_normalized, mfcc_max_normalized, mfcc_min_normalized], axis=1)


    return mfcc_df


def get_feature_mel(df):
    features = []
    for path in tqdm(df['path']):
        data, sr = librosa.load(path, sr=CFG['SR'])
        n_fft = 2048
        win_length = 2048
        hop_length = 1024
        n_mels = 128

        D = np.abs(librosa.stft(data, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
        mel = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)

        features.append({
            'mel_mean': mel.mean(axis=1),
            'mel_max': mel.min(axis=1),
            'mel_min': mel.max(axis=1),
        })

    mel_df = pd.DataFrame(features)
    mel_mean_df = pd.DataFrame(mel_df['mel_mean'].tolist(), columns=[f'mel_mean_{i}' for i in range(n_mels)])
    mel_max_df = pd.DataFrame(mel_df['mel_max'].tolist(), columns=[f'mel_max_{i}' for i in range(n_mels)])
    mel_min_df = pd.DataFrame(mel_df['mel_min'].tolist(), columns=[f'mel_min_{i}' for i in range(n_mels)])

    return pd.concat([mel_mean_df, mel_max_df, mel_min_df], axis=1)

train_mf = get_mfcc_feature(train_df)
test_mf = get_mfcc_feature(test_df)

train_mel = get_feature_mel(train_df)
test_mel = get_feature_mel(test_df)

train_x = pd.concat([train_mel, train_mf], axis=1)
test_x = pd.concat([test_mel, test_mf], axis=1)

train_y = train_df['label']

train_x['label'] = train_df['label']
train_data = TabularDataset(train_x)
test_data = TabularDataset(test_x)

label = 'label'
eval_metric = 'accuracy'
time_limit = 3600 * 1

predictor = TabularPredictor(
    label=label, eval_metric=eval_metric
).fit(train_data, presets='best_quality', time_limit=time_limit, ag_args_fit={'num_gpus': 0, 'num_cpus': 4})

model_to_use = predictor.get_model_best()
model_pred = predictor.predict(test_data, model=model_to_use)

submission = pd.read_csv('./_data/sample_submission.csv')
submission['label'] = model_pred
submission.to_csv('./autogluon_음성감정.csv', index=False)

# Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the -35.08s of remaining time.
        # 0.5733   = Validation score   (accuracy)
        # 0.56s    = Training   runtime
        # 0.0s     = Validation runtime
# AutoGluon training complete, total runtime = 3635.69s ... Best model: "WeightedEnsemble_L3"
# TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels\ag-20230526_062002\")

# Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the -26.29s of remaining time.
        # 0.5789   = Validation score   (accuracy)
        # 0.55s    = Training   runtime
        # 0.0s     = Validation runtime
# AutoGluon training complete, total runtime = 3626.87s ... Best model: "WeightedEnsemble_L3"
# TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels\ag-20230526_091414\")