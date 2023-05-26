#shape 오류인 것 내용 명시하고, 추가 모델 만들기 
#공동 FClayer구성하지 말고 GAP 바로 출력 

#1. VGG19
#2. Xception
#3. ResNet50
#4. ResNet101
#5. InceptionV3
#6. InceptionResNetV2
#7. DenseNet121
#8. MobileNetV2
#9. NASNetMobile
#10. EfficientNetB0

from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# model_list = [VGG19, Xception, ResNet50,ResNet101,InceptionV3,
#               InceptionResNetV2, DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0
#                ]
model_list = [VGG19, ResNet50, ResNet101,
              DenseNet121, MobileNetV2, EfficientNetB0]
# model = VGG16()
# model = VGG19()
#...

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train / 255.
x_test = x_test / 255.

for model in model_list:
    model_instance = model(weights='imagenet', include_top=False, input_shape=(32,32,3))
    model_instance.trainable = False

    # Create a new model
    new_model = Sequential()
    new_model.add(model_instance)
    new_model.add(GlobalAveragePooling2D())
    new_model.add(Dense(100, activation='softmax'))

    # Compile the model
    new_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    new_model.fit(x_train, y_train, batch_size=512, epochs=10, validation_data=(x_test, y_test), verbose=0,
                  callbacks=[EarlyStopping(patience=3), ReduceLROnPlateau(factor=0.2, patience=2)])

    #evaluate
    results = new_model.evaluate(x_test, y_test)
    print("================================")
    print("Model Name:", model.__name__)
    print("Total number of weights:", len(model_instance.weights))
    print("Total number of trainable weights:", len(model_instance.trainable_weights))
    print("loss:", results[0])
    print("acc:", results[1])

# ERROR======================#
# Xception : ValueError: Input size must be at least 71x71; Received: input_shape=(32, 32, 3)
# InceptionV3 : ValueError: Input size must be at least 75x75; Received: input_shape=(32, 32, 3)
# InceptionResNetV2 : ValueError: Input size must be at least 75x75; Received: input_shape=(32, 32, 3)
# NASNetMobile :  ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).  Received: input_shape=(32, 32, 3)

'''
결과 : model_instance.trainable = False
================================
Model Name: VGG19
Total number of weights: 32
Total number of trainable weights: 0
loss: 2.8584673404693604
acc: 0.30379998683929443
================================
Model Name: ResNet50
Total number of weights: 318
Total number of trainable weights: 0
loss: 4.135097503662109
acc: 0.08990000188350677
313/313 [==============================] - 14s 44ms/step - loss: 4.3969 - accuracy: 0.0515
================================
Model Name: ResNet101
Total number of weights: 624
Total number of trainable weights: 0
loss: 4.3969221115112305
acc: 0.051500000059604645
================================
Model Name: DenseNet121
Total number of weights: 604
Total number of trainable weights: 0
loss: 2.4050090312957764
acc: 0.400299996137619
WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
313/313 [==============================] - 7s 20ms/step - loss: 3.9900 - accuracy: 0.1224
================================
Model Name: MobileNetV2
Total number of weights: 260
Total number of trainable weights: 0
loss: 3.9899890422821045
acc: 0.12240000069141388
================================
Model Name: EfficientNetB0
Total number of weights: 312
Total number of trainable weights: 0
loss: 4.60508394241333
acc: 0.011599999852478504
================================

'''


'''
결과 : model_instance.trainable = True
================================
Model Name: VGG19
Total number of weights: 32
Total number of trainable weights: 32
loss: 2.6347715854644775
acc: 0.30379998683929443
313/313 [==============================] - 8s 26ms/step - loss: 10.7263 - accuracy: 0.0128
================================
Model Name: ResNet50
Total number of weights: 318
Total number of trainable weights: 212
loss: 10.726347923278809
acc: 0.012799999676644802
313/313 [==============================] - 14s 41ms/step - loss: 6.7974 - accuracy: 0.0204
================================
Model Name: ResNet101
Total number of weights: 624
Total number of trainable weights: 416
loss: 6.797378063201904
acc: 0.020400000736117363
313/313 [==============================] - 18s 55ms/step - loss: 1.6049 - accuracy: 0.6475
================================
Model Name: DenseNet121
Total number of weights: 604
Total number of trainable weights: 362
loss: 1.6049175262451172
acc: 0.6474999785423279
WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights 
for input shape (224, 224) will be loaded as the default.
313/313 [==============================] - 7s 22ms/step - loss: 17.7545 - accuracy: 0.0174 
================================
Model Name: MobileNetV2
Total number of weights: 260
Total number of trainable weights: 156
loss: 17.754547119140625
acc: 0.017400000244379044
313/313 [==============================] - 10s 32ms/step - loss: 6.1837 - accuracy: 0.0098
================================
Model Name: EfficientNetB0
Total number of weights: 312
Total number of trainable weights: 211
loss: 6.183740615844727
acc: 0.009800000116229057
'''