# 0. seed
import random
import numpy as np
import tensorflow as tf
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. 데이터
import pandas as pd
path="./_data/dacon_wine/"
path_save='./_save/dacon_wine/'
df=pd.read_csv(path+'train.csv',index_col=0)
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'sample_submission.csv')

from sklearn.preprocessing import MinMaxScaler,LabelEncoder,RobustScaler
le=LabelEncoder()
df[df.columns[-1]]=le.fit_transform(df[df.columns[-1]])
dft[df.columns[-1]]=le.transform(dft[df.columns[-1]])

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
imputer=IterativeImputer(XGBRegressor())
df=pd.DataFrame(imputer.fit_transform(df),columns=df.columns,index=df.index)
df=df.interpolate().dropna()
x=df.drop([df.columns[0]], axis=1)
y=df[df.columns[0]]
y-=np.min(y)
print(np.unique(y,return_counts=True))
unique_classes, counts = np.unique(y, return_counts=True)

# 클래스 레이블을 정수형으로 변환합니다.
unique_classes = unique_classes.astype(int)

# 가장 많은 수를 가진 클래스와 그 수를 찾습니다.
max_class = unique_classes[np.argmax(counts)]
max_count = np.max(counts)

# 데이터를 오버샘플링하여 새로운 데이터프레임에 저장합니다.
x_resampled = pd.DataFrame()
y_resampled = pd.Series()

for cls in unique_classes:
    # 현재 클래스의 인덱스를 찾습니다.
    idx = y[y == cls].index
    
    # 현재 클래스의 샘플 수를 가장 많은 클래스의 샘플 수로 맞추기 위해 필요한 복사 횟수를 계산합니다.
    n_samples = int(max_count / counts[cls]) - 1
    
    # 현재 클래스의 샘플들을 복사하여 오버샘플링합니다.
    x_class_resampled = np.repeat(x.loc[idx].values, repeats=n_samples, axis=0)
    y_class_resampled = np.repeat(y[idx].values, repeats=n_samples, axis=0)
    
    # 새로운 데이터프레임에 추가합니다.
    x_resampled = x_resampled.append(pd.DataFrame(x_class_resampled, columns=x.columns), ignore_index=True)
    y_resampled = y_resampled.append(pd.Series(y_class_resampled, name=y.name), ignore_index=True)

# 원래 데이터와 오버샘플링된 데이터를 합칩니다.
x_resampled = x_resampled.append(x, ignore_index=True)
y_resampled = y_resampled.append(y, ignore_index=True)
x=x_resampled
y=y_resampled
y=np.array(pd.get_dummies(y,prefix='number'))
print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)

scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
dft=scaler.transform(dft)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout,LeakyReLU, BatchNormalization
model=Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Dense(512,activation=LeakyReLU(0.6),use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(1/32))
model.add(Dense(512,activation=LeakyReLU(0.6),use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(1/32))
model.add(Dense(512,activation=LeakyReLU(0.6),use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(1/32))
model.add(Dense(512,activation=LeakyReLU(0.6),use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(1/32))
model.add(Dense(512,activation=LeakyReLU(0.6),use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(1/32))
model.add(Dense(y_train.shape[1],activation='softmax'))

# 3. compile,training
from tensorflow.keras.callbacks import EarlyStopping as es
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10000,batch_size=len(x_train)//99,
          callbacks=es(monitor='val_acc',mode='max',patience=20,restore_best_weights=True,verbose=True))

y_pred = model.predict(dft)

submission = pd.read_csv('./_data/dacon_wine/sample_submission.csv', index_col=0)


# print(y_pred)
# print(np.argmax(y_pred))
submission['quality'] = np.argmax(y_pred, axis=1)+3

print(submission)

submission.to_csv('./_data/dacon_wine/sub1.csv')