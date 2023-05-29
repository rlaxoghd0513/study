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
    'SR': 22000,  # 높으면 잘 잘라줌
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
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=2048, hop_length=512)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
        pitch, _ = librosa.piptrack(y=y, sr=sr, S=None, n_fft=2048, hop_length=512)

        features.append({
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_max': np.max(mfcc, axis=1),
            'mfcc_min': np.min(mfcc, axis=1),
            'spectral_contrast_mean': np.mean(spectral_contrast, axis=1),
            'spectral_contrast_max': np.max(spectral_contrast, axis=1),
            'spectral_contrast_min': np.min(spectral_contrast, axis=1),
            'chroma_mean': np.mean(chroma, axis=1),
            'chroma_max': np.max(chroma, axis=1),
            'chroma_min': np.min(chroma, axis=1),
            'pitch_mean': np.mean(pitch, axis=1),
            'pitch_max': np.max(pitch, axis=1),
            'pitch_min': np.min(pitch, axis=1),
        })

    features_df = pd.DataFrame(features)
    mfcc_mean_df = pd.DataFrame(features_df['mfcc_mean'].tolist(), columns=[f'mfcc_mean_{i}' for i in range(CFG['N_MFCC'])])
    mfcc_max_df = pd.DataFrame(features_df['mfcc_max'].tolist(), columns=[f'mfcc_max_{i}' for i in range(CFG['N_MFCC'])])
    mfcc_min_df = pd.DataFrame(features_df['mfcc_min'].tolist(), columns=[f'mfcc_min_{i}' for i in range(CFG['N_MFCC'])])
    spectral_contrast_mean_df = pd.DataFrame(features_df['spectral_contrast_mean'].tolist(), columns=[f'spectral_contrast_mean_{i}' for i in range(len(features_df['spectral_contrast_mean'][0]))])
    spectral_contrast_max_df = pd.DataFrame(features_df['spectral_contrast_max'].tolist(), columns=[f'spectral_contrast_max_{i}' for i in range(len(features_df['spectral_contrast_max'][0]))])
    spectral_contrast_min_df = pd.DataFrame(features_df['spectral_contrast_min'].tolist(), columns=[f'spectral_contrast_min_{i}' for i in range(len(features_df['spectral_contrast_min'][0]))])
    chroma_mean_df = pd.DataFrame(features_df['chroma_mean'].tolist(), columns=[f'chroma_mean_{i}' for i in range(len(features_df['chroma_mean'][0]))])
    chroma_max_df = pd.DataFrame(features_df['chroma_max'].tolist(), columns=[f'chroma_max_{i}' for i in range(len(features_df['chroma_max'][0]))])
    chroma_min_df = pd.DataFrame(features_df['chroma_min'].tolist(), columns=[f'chroma_min_{i}' for i in range(len(features_df['chroma_min'][0]))])
    pitch_mean_df = pd.DataFrame(features_df['pitch_mean'].tolist(), columns=[f'pitch_mean_{i}' for i in range(len(features_df['pitch_mean'][0]))])
    pitch_max_df = pd.DataFrame(features_df['pitch_max'].tolist(), columns=[f'pitch_max_{i}' for i in range(len(features_df['pitch_max'][0]))])
    pitch_min_df = pd.DataFrame(features_df['pitch_min'].tolist(), columns=[f'pitch_min_{i}' for i in range(len(features_df['pitch_min'][0]))])

    mfcc_mean_normalized = pd.DataFrame(scaler.fit_transform(mfcc_mean_df), columns=[f'mfcc_mean_norm_{i}' for i in range(CFG['N_MFCC'])])  # 정규화 추가
    mfcc_max_normalized = pd.DataFrame(scaler.fit_transform(mfcc_max_df), columns=[f'mfcc_max_norm_{i}' for i in range(CFG['N_MFCC'])])  # 정규화 추가
    mfcc_min_normalized = pd.DataFrame(scaler.fit_transform(mfcc_min_df), columns=[f'mfcc_min_norm_{i}' for i in range(CFG['N_MFCC'])])  # 정규화 추가
    spectral_contrast_mean_normalized = pd.DataFrame(scaler.fit_transform(spectral_contrast_mean_df), columns=[f'spectral_contrast_mean_norm_{i}' for i in range(len(spectral_contrast_mean_df.columns))])
    spectral_contrast_max_normalized = pd.DataFrame(scaler.fit_transform(spectral_contrast_max_df), columns=[f'spectral_contrast_max_norm_{i}' for i in range(len(spectral_contrast_max_df.columns))])
    spectral_contrast_min_normalized = pd.DataFrame(scaler.fit_transform(spectral_contrast_min_df), columns=[f'spectral_contrast_min_norm_{i}' for i in range(len(spectral_contrast_min_df.columns))])
    chroma_mean_normalized = pd.DataFrame(scaler.fit_transform(chroma_mean_df), columns=[f'chroma_mean_norm_{i}' for i in range(len(chroma_mean_df.columns))])
    chroma_max_normalized = pd.DataFrame(scaler.fit_transform(chroma_max_df), columns=[f'chroma_max_norm_{i}' for i in range(len(chroma_max_df.columns))])
    chroma_min_normalized = pd.DataFrame(scaler.fit_transform(chroma_min_df), columns=[f'chroma_min_norm_{i}' for i in range(len(chroma_min_df.columns))])
    pitch_mean_normalized = pd.DataFrame(scaler.fit_transform(pitch_mean_df), columns=[f'pitch_mean_norm_{i}' for i in range(len(pitch_mean_df.columns))])
    pitch_max_normalized = pd.DataFrame(scaler.fit_transform(pitch_max_df), columns=[f'pitch_max_norm_{i}' for i in range(len(pitch_max_df.columns))])
    pitch_min_normalized = pd.DataFrame(scaler.fit_transform(pitch_min_df), columns=[f'pitch_min_norm_{i}' for i in range(len(pitch_min_df.columns))])

    feature_df = pd.concat([mfcc_mean_normalized, mfcc_max_normalized, mfcc_min_normalized,
                           spectral_contrast_mean_normalized, spectral_contrast_max_normalized, spectral_contrast_min_normalized,
                           chroma_mean_normalized, chroma_max_normalized, chroma_min_normalized,
                           pitch_mean_normalized, pitch_max_normalized, pitch_min_normalized], axis=1)

    return feature_df


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

submission = pd.read_csv('./_data/dacon_voice/sample_submission.csv')
submission['label'] = model_pred

submission.to_csv('./_save/dacon_voice/submission.csv', index=False)
