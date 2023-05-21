import autokeras as ak
from keras.datasets import cifar10

# CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# AutoKeras 이미지 분류 모델 생성
model = ak.ImageClassifier(max_trials=1, overwrite=False)  # 최대 시도 횟수 지정

# 모델 훈련
model.fit(x_train, y_train, epochs=10)

# 모델 평가
results = model.evaluate(x_test, y_test)
print('결과:', results)

# 결과: [0.8242470622062683, 0.7192999720573425]