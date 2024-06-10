#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define class names
CLASS_NAMES = ['rock', 'paper', 'scissors']

# Load the model
model = tf.keras.models.load_model('rock_paper_scissors_cnn.h5')

# Load the image from the supplied argument
image_path = sys.argv[1]
image = Image.open(image_path)
image = image.convert('RGB')

# Preprocess the image
image = image.resize((150, 150))
image = np.array(image)
image = image / 255.0
image = np.expand_dims(image, axis=0)

# Make a prediction
predictions = model.predict(image)
class_index = np.argmax(predictions[0])
class_name = CLASS_NAMES[class_index]
score = predictions[0][class_index]

# Visualize the image with the predicted label and score
plt.imshow(image[0])
plt.axis('off')
plt.title(f'Prediction: {class_name}, Score: {score:.2f}')
plt.show()


# In[ ]:




