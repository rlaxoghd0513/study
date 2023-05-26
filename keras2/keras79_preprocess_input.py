from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

#가중치 사용 방법 
model = ResNet50(weights='imagenet')    #imagenet가중치 사용
# model = ResNet50(weights=None)        #가중치 사용x
# model = ResNet50(weights='경로')      #내가 만든 가중치 


# path = 'D:\study\_data\cat_dog\PetImages\Dog\\4.jpg'
# path = 'D:\study\_data\suit.png'
path = 'D:\study\_data\pmk.jpg'

img = image.load_img(path, target_size=(224,224))
print(img) #<PIL.Image.Image image mode=RGB size=224x224 at 0x1F88F30D070>

# 이미지 변환(이미지 수치화 )
x = image.img_to_array(img) # 이미지를 x에 집어넣음 
print("===================== image.img_to_array(img) =======================")
# print(x)
print(x.shape)  # (224, 224, 3)  // 그림 수치화 완료 
print(np.min(x), np.max(x)) # 0.0 255.0   //이미지 : 0~255사이 

# x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
# # x = x.reshape(1, *x.shape)
# print(x.shape)     #(1, 224, 224, 3)

x = np.expand_dims(x, axis=0) #첫번째 축 늘리기 (reshape랑 동일)
print(x.shape)       #  (1, 224, 224, 3)

######################## -155 에서 155 사이로 정규화 ###############################
print("===================== preprocess_input(x) =======================")
x = preprocess_input(x)
print(x.shape)
print(np.min(x), np.max(x)) #-123.68 151.061

print("===================== model.predict(x) ===========================")
x_pred = model.predict(x)
# model.summary()
print(x_pred, '\n', x_pred.shape)

print("결과 :", decode_predictions(x_pred, top=5))


'''
===================== image.img_to_array(img) =======================
(224, 224, 3)
0.0 255.0
(1, 224, 224, 3)
===================== preprocess_input(x) =======================
(1, 224, 224, 3)
-123.68 151.061
===================== model.predict(x) ===========================
 (1, 1000)

 결과 : 
 [[('n02093428', 'American_Staffordshire_terrier', 0.24548924), 
 ('n02099712', 'Labrador_retriever', 0.087651126), 
 ('n02088364', 'beagle', 0.078063615), 
 ('n02085620', 'Chihuahua', 0.060257856), 
 ('n04409515', 'tennis_ball', 0.051135786)]]

'''

# pmk결과 :
#  [[('n04584207', 'wig', 0.68206567), 
# ('n03630383', 'lab_coat', 0.06610236), 
# ('n03595614', 'jersey', 0.032460064), 
# ('n04317175', 'stethoscope', 0.029964073), 
# ('n03877472', 'pajama', 0.013030323)]]