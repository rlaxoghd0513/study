from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception


model_list = [VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2,
              DenseNet121, DenseNet169, DenseNet201, InceptionV3, InceptionResNetV2,
              MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
              NASNetMobile, NASNetLarge, EfficientNetB0, EfficientNetB1, EfficientNetB7, Xception]
# model = VGG16()
# model = VGG19()
#...


for model in model_list:
    model = model(weights='imagenet', include_top=False)
    model.trainable = False
    #model.summary()    
    print("================================")
    print("모델명:", model.__name__)
    print("전체 가중치 개수:", len(model.weights))
    print("전체 훈련가능 개수:", len(model.trainable_weights))


#결과 출력 