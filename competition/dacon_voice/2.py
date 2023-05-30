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
import optuna

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')

CFG = {
    'SR': 22100,
    'N_MFCC': 128,
    'N_MELS': 769,  # Update the number of mel frequency coefficients
    'SEED': 42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED']) # Seed 고정


train_df = pd.read_csv('./_data/dacon_voice/train.csv')
test_df = pd.read_csv('./_data/dacon_voice/test.csv')

train_df['path'] = './_data/dacon_voice/' + train_df['path']
test_df['path'] = './_data/dacon_voice/' + test_df['path']

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

    return pd.concat([mfcc_mean_df, mfcc_max_df, mfcc_min_df], axis=1)


def get_feature_mel(df):
    features = []
    for path in tqdm(df['path']):
        data, sr = librosa.load(path, sr=CFG['SR'])
        n_fft = 2048
        win_length = 2048
        hop_length = 1024
        n_mels = 128

        D = np.abs(librosa.stft(data, n_fft=n_fft, win_length = win_length, hop_length=hop_length))
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
train_x = TabularDataset(train_x)
test_x = TabularDataset(test_x)
train_x = train_x.drop('label', axis=1)

# Split data into training and validation sets
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=CFG['SEED'])

def objective(trial):
    # Hyperparameter search space and XGBoost parameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 800, 900, step=10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.1, 0.9),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
        'random_state': CFG['SEED']
    }

    model = XGBClassifier(**params)
    model.fit(train_x, train_y)

    preds = model.predict(val_x)
    accuracy = accuracy_score(val_y, preds)
    return accuracy
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1)
best_params = study.best_params
best_params['random_state'] = CFG['SEED']
model = XGBClassifier(**best_params)
model.fit(train_x, train_y)
preds = model.predict(test_x)
preds_acc = model.predict(train_x)
submission = pd.read_csv('./_data/dacon_voice/sample_submission.csv')
submission['label'] = preds
# DATE
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
submission.to_csv(date + 'TTS_XGB_Three_submission.csv', index=False)