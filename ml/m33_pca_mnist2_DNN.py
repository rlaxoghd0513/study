import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()   #y가져오기 싫다 파이썬 기초문법 _작대기

n_components_list = [154,331,486,713]
list_name = ['0.95','0.99','0.999','1.0']

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

    model = Sequential() 
    model.add(Dense(16, input_dim=n_components_list[i]))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
    model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split = 0.2)
    
    acc = model.evaluate(x_test, y_test)[1]
    print('pca',list_name[i],':', acc)

# 최고DNN :  0.9223
# 최고CNN :  0.9814
# pca 0.95 : 0.8677856922149658
# pca 0.99 : 0.8637857437133789
# pca 0.999 : 0.8684285879135132
# pca 1.0 : 0.8687142729759216


