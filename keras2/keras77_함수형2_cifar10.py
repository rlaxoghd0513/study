from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.datasets import cifar10

base_model = VGG16(weights = 'imagenet', include_top=False, input_shape = (32,32,3))

print(base_model.output)
# KerasTensor(type_spec=TensorSpec(shape=(None, None, Nonem 512),...............))

x = base_model.output
x = GlobalAveragePooling2D()(x)
output1 = Dense(10, activation='softmax')(x)

model = Model(inputs = base_model.input, outputs = output1)
#base_model의 인풋을 받고 output을 averagepooling한다

model.summary()