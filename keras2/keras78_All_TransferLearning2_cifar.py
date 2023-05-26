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
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model_list = [DenseNet169, DenseNet201, InceptionV3, InceptionResNetV2,
              MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
              NASNetMobile, NASNetLarge, EfficientNetB0, EfficientNetB1, EfficientNetB7, Xception]

# model_list = [VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2,
#               DenseNet121, DenseNet169, DenseNet201, InceptionV3, InceptionResNetV2,
#               MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
#               NASNetMobile, NASNetLarge, EfficientNetB0, EfficientNetB1, EfficientNetB7, Xception]
# model = VGG16()
# model = VGG19()
#...

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
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
    new_model.add(Dense(10, activation='softmax'))

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



#결과 출력 
'''
================================
Model Name: VGG16
Total number of weights: 26
Total number of trainable weights: 0
loss: 1.3018743991851807
acc: 0.5526000261306763
313/313 [==============================] - 3s 10ms/step - loss: 1.3211 - accuracy: 0.5443
================================
Model Name: VGG19
Total number of weights: 32
Total number of trainable weights: 0
loss: 1.3210654258728027
acc: 0.5443000197410583
313/313 [==============================] - 8s 26ms/step - loss: 1.8408 - accuracy: 0.3573
================================
Model Name: ResNet50
Total number of weights: 318
Total number of trainable weights: 0
loss: 1.840796709060669
acc: 0.3573000133037567
313/313 [==============================] - 8s 26ms/step - loss: 1.8147 - accuracy: 0.3819
================================
Model Name: ResNet50V2
Total number of weights: 270
Total number of trainable weights: 0
loss: 1.8146743774414062
acc: 0.38190001249313354
313/313 [==============================] - 15s 47ms/step - loss: 2.0785 - accuracy: 0.2598
================================
Model Name: ResNet101
Total number of weights: 624
Total number of trainable weights: 0
loss: 2.078450918197632
acc: 0.259799987077713
313/313 [==============================] - 18s 58ms/step - loss: 1.7844 - accuracy: 0.3826
================================
Model Name: ResNet101V2
Total number of weights: 542
Total number of trainable weights: 0
loss: 1.784403920173645
acc: 0.38260000944137573
================================
Model Name: ResNet152
Total number of weights: 930
Total number of trainable weights: 0
loss: 2.1034951210021973
acc: 0.2524000108242035
313/313 [==============================] - 21s 67ms/step - loss: 1.8509 - accuracy: 0.3685
================================
Model Name: ResNet152V2
Total number of weights: 814
Total number of trainable weights: 0
loss: 1.850934624671936
acc: 0.3684999942779541
313/313 [==============================] - 18s 53ms/step - loss: 1.0686 - accuracy: 0.6310
================================
Model Name: DenseNet121
Total number of weights: 604
Total number of trainable weights: 0
loss: 1.0686105489730835
acc: 0.6309999823570251
================================
Model Name: DenseNet169
Total number of weights: 844
Total number of trainable weights: 0
loss: 1.0936312675476074
acc: 0.6245999932289124
313/313 [==============================] - 26s 80ms/step - loss: 1.1044 - accuracy: 0.6236
================================
Model Name: DenseNet201
Total number of weights: 1004
Total number of trainable weights: 0
loss: 1.1044340133666992
acc: 0.6236000061035156