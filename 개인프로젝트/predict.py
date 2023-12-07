#예측
Predict_path = 'd:/study_data/_data/project/project/Predict/'
save_path_predict = 'd:/study_data/_save/project/predict/'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

datagen = ImageDataGenerator(rescale=1./255)

predict_data = datagen.flow_from_directory(Predict_path,
                                  target_size = (150,150),
                                  batch_size = 7, 
                                  class_mode = 'categorical',
                                  color_mode = 'rgb',
                                  shuffle=False)

x_predict = predict_data[0][0]
y_predict = predict_data[0][1]

np.save(save_path_predict + 'x_predict.npy', arr = x_predict)
np.save(save_path_predict + 'y_predict.npy', arr = y_predict)
