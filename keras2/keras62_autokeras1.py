import autokeras as ak
from keras.datasets import mnist
import time
# print(ak.__version__) #1.0.20

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

# autokeras 모델 생성
model = ak.ImageClassifier(max_trials=2,
                         overwrite=False)  # 최대 시도 횟수 지정

# 모델 훈련
start = time.time()
model.fit(x_train, y_train, epochs=2, validation_split=0.15)  # 적절한 epoch 수를 지정하여 훈련합니다.
end = time.time()

# 모델 평가
y_predict = model.predict(x_test)

results = model.evaluate(x_test,y_test)
print('결과 :', results)
print('걸린시간 :', round(end-start, 4))


#5. 최적의 모델 출력
best_model = model.export_model()
print(best_model.summary())

# 최적의 모델 저장
path = './_save/autokeras/'
best_model.save(path + "keras62_autokeras1.h5")
# 결과 : [0.036897897720336914, 0.9894000291824341]
# 걸린시간 : 1293.8924

# 결과 : [0.030224395915865898, 0.991599977016449]
# 걸린시간 : 62.0727
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 28, 28)]          0

#  cast_to_float32 (CastToFloa  (None, 28, 28)           0
#  t32)

#  expand_last_dim (ExpandLast  (None, 28, 28, 1)        0
#  Dim)

#  normalization (Normalizatio  (None, 28, 28, 1)        3
#  n)

#  conv2d (Conv2D)             (None, 26, 26, 32)        320

#  conv2d_1 (Conv2D)           (None, 24, 24, 64)        18496

#  max_pooling2d (MaxPooling2D  (None, 12, 12, 64)       0
#  )

#  dropout (Dropout)           (None, 12, 12, 64)        0

#  flatten (Flatten)           (None, 9216)              0

#  dropout_1 (Dropout)         (None, 9216)              0

#  dense (Dense)               (None, 10)                92170

#  classification_head_1 (Soft  (None, 10)               0
#  max)

# =================================================================
# Total params: 110,989
# Trainable params: 110,986
# Non-trainable params: 3
# _________________________________________________________________
# None