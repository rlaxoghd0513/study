import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()   #y가져오기 싫다 파이썬 기초문법 _작대기

x = np.append(x_train, x_test, axis=0)
print(x.shape) 
y = np.append(y_train, y_test, axis=0)

x = x.reshape(70000,784)
n_components_list = [154,331,486,713]
list_name = ['0.95','0.99','0.999','1.0']

print('나의 최고의 CNN: 0.9814')
print('나의 최고의 DNN: 0.9223')

for i,value in enumerate(n_components_list):
    pca = PCA(n_components = value)
    x = pca.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=123, shuffle=True)

    model = RandomForestClassifier(random_state=123) #훈련할때마다 바뀌니까 random_state고정

    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print('pca',list_name[i],':', results)
    pca = PCA(n_components = 784)


