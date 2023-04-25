import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from xgboost import XGBClassifier

(x_train, y_train), (x_test, y_test) = mnist.load_data() 

n_components_list = [154,331,486,713]
list_name = ['0.95','0.99','0.999','1.0']
parameters = parameters = [
    {"n_estimators": [100, 200, 300],
     "learning_rate": [0.1, 0.3, 0.001, 0.01],
    "max_depth": [4, 5, 6]},

    {"n_estimators": [90, 100, 110],
    "learning_rate": [0.1, 0.001, 0.01],
    "max _depth": [4,5,6],
    "colsample_bytree": [0.6, 0.9, 1]},

    {"n_estimators": [90, 110],
    "learning rate": [0.1, 0.001, 0.5],
    "max _depth": [4,5,6],
    "colsample _bytree": [0.6, 0.9, 1]},

    {"colsample_bylevel": [0.6, 0.7, 0.9]}]


for i,value in enumerate(n_components_list):
    (x_train, y_train), (x_test, y_test) = mnist.load_data() 

    x = np.append(x_train, x_test, axis=0)
    print(x.shape) 
    y = np.append(y_train, y_test, axis=0)

    x = x.reshape(70000,784)
    y = to_categorical(y)
    
    pca = PCA(n_components = value)
    x = pca.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=123, shuffle=True)

    model = GridSearchCV(XGBClassifier(tree_method ='gpu_hist',predictor ='gpu_predictor',gpu_id =0), 
                         parameters, cv=5,verbose=1,n_jobs=-1)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    
    
    print('pca',list_name[i],':', result)
    
# pca 0.95 : 0.9365714285714286
# pca 0.99 : 0.9305
# pca 0.999 : 0.9300714285714285
# pca 1.0 : 0.9252857142857143
    
    