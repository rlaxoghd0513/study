import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from xgboost import XGBClassifier
import time
# Load data
train = pd.read_csv('./_data/dacon_air/train.csv')
test = pd.read_csv('./_data/dacon_air/test.csv')
sample_submission = pd.read_csv('./_data/dacon_air/sample_submission.csv', index_col=0)

#print(train)
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

for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train[i])
    train[i] = le.transform(train[i])

    for label in np.unique(test[i]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    test[i] = le.transform(test[i])
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
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state= 11)

# Normalize numerical features
# scaler = StandardScaler()
# train_x = scaler.fit_transform(train_x)
# val_x = scaler.transform(val_x)
# test_x = scaler.transform(test_x)

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import datetime

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)

# Set up CatBoost classifier
model = CatBoostClassifier(random_state=11,
                           task_type='GPU',
                           devices='0',
                           learning_rate=0.05,
                           depth=4,
                           iterations=800,
                           verbose=0)

# Set up hyperparameter grid
# param_grid = {
#     'learning_rate': [0.05, 0.003],
#     'depth': [2, 4],
#     'iterations': [600, 800],
# }dacon_air

# # Set up grid search
# grid = GridSearchCV(model,
#                     param_grid,
#                     cv=cv,
#                     scoring='accuracy',
#                     n_jobs=-1,
#                     verbose=1)

# Fit the model and find the best hyperparameters
model.fit(train_x, train_y)

# Use the model to make predictions on the test set
y_pred = model.predict_proba(test_x)

# Save the predictions
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

save_path = './_save/dacon_air/'
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv(save_path + date + '_sample_submission.csv', index=True)
