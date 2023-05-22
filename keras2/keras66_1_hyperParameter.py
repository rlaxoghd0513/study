import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, -1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32')/255.

# print(x_test.shape)

#2. 모델
def build_model(drop=0.5, optimizer= 'adam',activation='relu', 
                node1=64, node2=64, node3=64, lr=0.001):
    inputs = Input(shape=(784), name= 'inputs')
    x = Dense(512, activation=activation, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name = 'hidden3')(x)
    x = Dropout(drop)(x)    
    x = Dense(256, activation=activation, name = 'hidden4')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name = 'outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss = 'sparse_categorical_crossentropy')
    return model

def create_hyperparameter():
    batchs =[100, 200, 300, 400, 500]
    optimizer = ['adam', 'rmsprop', 'adadlta']
    dropout = [0.2, 0.3, 0.4, 0.5]
    activation = ['relu', 'elu', 'selu', 'linear']
    return{'batch_size' : batchs,
        'optimizer' : optimizer,
        'drop' : dropout,
        'activation' : activation}

hyperparameters = create_hyperparameter()
# print(hyperparameter) 
# {'batch_size': [100, 200, 300, 400, 500], 'optimizer': ['adam', 'rmsprop', 'adadlta'], 
# 'dropout': [0.2, 0.3, 0.4, 0.5], 'activation': ['relu', 'elu', 'selu', 'linear']}
from sklearn.model_selection import GridSearchCV
estimator = KerasClassifier(build_fn=build_model, epochs=10, batch_size=32)

grid = GridSearchCV(estimator=estimator, param_grid=hyperparameters, cv=5)

grid_result = grid.fit(x_train,y_train)

# 최적 파라미터와 최적 점수 출력
print("Best Score: ", grid_result.best_score_)
print("Best Params: ", grid_result.best_params_)
