import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from xgboost import XGBClassifier
import time
# Load data
train = pd.read_csv('./_data/dacon_air/train.csv')
test = pd.read_csv('./_data/dacon_air/test.csv')
sample_submission = pd.read_csv('./_data/dacon_air/sample_submission.csv', index_col=0)

# Load data
train = pd.read_csv('./_data/dacon_air/train.csv')
test = pd.read_csv('./_data/dacon_air/test.csv')
sample_submission = pd.read_csv('./_data/dacon_air/sample_submission.csv', index_col=0)

# Replace variables with missing values except for the label (Delay) with the most frequent values of the training data
NaN = ['Origin_State', 'Destination_State', 'Airline', 'Estimated_Departure_Time', 'Estimated_Arrival_Time', 'Carrier_Code(IATA)', 'Carrier_ID(DOT)']

for col in NaN:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)

    if col in test.columns:
        test[col] = test[col].fillna(mode)
print('Done.')

# Quantify qualitative variables
qual_col = ['Origin_Airport', 'Origin_State', 'Destination_Airport', 'Destination_State', 'Airline', 'Carrier_Code(IATA)', 'Tail_Number']

train = pd.get_dummies(train, columns=qual_col)
test = pd.get_dummies(test, columns=qual_col)

# Align train and test set
train, test = train.align(test, join='inner', axis=1)
print('Done.')

# Remove unlabeled data
train = train.dropna()

column4 = {}
for i, column in enumerate(sample_submission.columns):
    column4[column] = i

def to_number(x, dic):
    return dic[x]

train.loc[:, 'Delay_num'] = train['Delay'].apply(lambda x: to_number(x, column4))
print('Done.')

train_x = train.drop(columns=['ID', 'Delay', 'Delay_num'])
train_y = train['Delay_num']
test_x = test.drop(columns=['ID'])

pf = PolynomialFeatures(degree=2)
train_x = pf.fit_transform(train_x)

# Split the training dataset into a training set and a validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=123)

# Normalize numerical features
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

# Cross-validation with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# Model and hyperparameter tuning using GridSearchCV
model = XGBClassifier(random_state=55,
                      tree_method='gpu_hist', 
                      gpu_id=0, 
                      predictor = 'gpu_predictor')

param_grid = {
    'learning_rate': [0.03, 0.05],
    'max_depth': [2,5],
    'n_estimators': [300, 500],
}

grid = RandomizedSearchCV(model,
                    param_grid,
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1)

grid.fit(train_x, train_y)

best_model = grid.best_estimator_

# Model evaluation
val_y_pred = best_model.predict(val_x)

acc = accuracy_score(val_y, val_y_pred)
f1 = f1_score(val_y, val_y_pred, average='weighted')
pre = precision_score(val_y, val_y_pred, average='weighted')
recall = recall_score(val_y, val_y_pred, average='weighted')

print('Accuracy_score:', acc)
print('F1 Score:', f1)
print('Precision Score:', pre)
print('Recall Score:', recall)

y_pred = best_model.predict_proba(test_x)
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('./_save/dacon_air/1031submission.csv', float_format='%.3f')