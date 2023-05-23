#loss와 weight의 관계를 그려라
#loss는 pickle에 저장되있고 weight는 model에 저장되있음

import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

# Load the saved history object
with open('./_save/keras70_1_mnist_graph.pkl', 'rb') as f:
    history = pickle.load(f)

model = load_model('./_save/keras70_1_mnist_graph.h5')
weights = model.get_weights()

# Get the loss values
loss = history['loss']

# Plot the changes in loss for each weight parameter
# for i, w in enumerate(weights):
#     weight_values = w.flatten()[:len(loss)]  # 손실과 크기를 맞추기 위해 가중치 값 잘라냄
#     plt.plot(weight_values, loss, label=f'Weight {i+1}')


# plt.title('Weight vs. Loss')
# plt.xlabel('Weight')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# Reshape the weights array for plotting
import numpy as np
weights_flat = np.concatenate([w.flatten()[:len(loss)] for w in weights])

# Trim the weights_flat array to match the length of loss
weights_flat = weights_flat[:len(loss)]

# Plot the changes in loss for each weight parameter
plt.scatter(weights_flat, loss, marker='.', c='red')

plt.title('Weight vs. Loss')
plt.xlabel('Weight')
plt.ylabel('Loss')
plt.show()