path = 'd:/study/_save/'

import numpy as np

x = np.load(path+'증사_x.npy')
y = np.load(path+'증사_y.npy')

x_noised = x + np.random.normal(0, 0.2, size = x.shape) # 0에서 0.1사이의 값을 랜덤하게 넣어준다

x_noised = np.clip(x_noised, a_min = 0, a_max=1) #0보다 작은건 0으로 1보다 큰건 1로 고정시켜준다

print(np.max(x_noised), np.min(x_noised)) #1.0 0.0


#####아까 만든거 가지고 확인####
#2 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D#맥스풀링반대

def autoencoder():
    model = Sequential()
    #인코더
    model.add(Conv2D(32,(3,3), activation='relu', padding='same', input_shape=(200,200,3)))
    model.add(MaxPooling2D()) #디폴트가 (2,2)
    model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D()) #(n,7,7,8) 
    #디코더
    model.add(Conv2D(128,(3,3), activation='relu', padding='same'))
    model.add(UpSampling2D()) #(n,14,14,8)
    model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
    model.add(UpSampling2D()) #(n,28,28,16)
    model.add(Conv2D(3, (3,3), activation='sigmoid', padding='same'))
    return model

model = autoencoder()

#3 컴파일 훈련
model.compile(optimizer = 'adam', loss='mse')
model.fit(x_noised, x, epochs=1000, batch_size=16)

#4 평가 예측
decoded_images =model.predict(x_noised)

###################################################################################################

import matplotlib.pyplot as plt
if x.shape[0] == 1:
    x = x[0]
    x_noised = x_noised[0]
    decoded_images = decoded_images[0]

# Display the images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Plot the original image
axes[0].imshow(x, cmap='cool')
axes[0].set_title("Original Image")
axes[0].axis("off")

# Plot the noisy image
axes[1].imshow(x_noised, cmap='cool')
axes[1].set_title("x_noised")
axes[1].axis("off")

# Plot the decoded image
axes[2].imshow(decoded_images, cmap='cool')
axes[2].set_title("Decoded Image")
axes[2].axis("off")

# Adjust the layout and display the figure
plt.tight_layout()
plt.show()






