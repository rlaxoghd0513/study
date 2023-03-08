from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
#1.데이터

datasets = load_boston()
x = datasets.data
y = datasets['target'] #x,y 구조 동일
print(x.shape, y.shape) #(506, 13) (506,)
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=123, test_size=0.3)

#2.모델구성
model=Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
hist = model.fit(x_train, y_train, epochs=20, batch_size=8, validation_split=0.2, verbose=1)
print(hist.history)

#4. 
