import autokeras as ak
from keras.datasets import mnist
import time
import tensorflow as tf
from keras.utils.np_utils import to_categorical
# print(ak.__version__) #1.0.20

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# autokeras 모델 생성
model = ak.ImageClassifier(max_trials=2,
                         overwrite=False)  # 최대 시도 횟수 지정


path = './_save/autokeras/'
model = tf.keras.models.load_model(path + "keras62_autokeras1.h5")

# 모델 평가
y_predict = model.predict(x_test)
results = model.evaluate(x_test,y_test)
print('결과 :', results)
