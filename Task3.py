#!/usr/bin/env python
# coding: utf-8

# In[10]:


import sys
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('rock_paper_scissors_cnn.h5')

# Define the label mapping
labels = {0: 'rock', 1: 'paper', 2: 'scissors'}

# Load the two input images
img1 = Image.open(sys.argv[1]).convert('RGB')
img2 = Image.open(sys.argv[2]).convert('RGB')

# Resize the images to the input size of the model
img1 = img1.resize((150, 150))
img2 = img2.resize((150, 150))

# Convert the images to arrays and normalize the pixel values
x1 = np.asarray(img1) / 255.0
x2 = np.asarray(img2) / 255.0

# Add a batch dimension to the arrays
x1 = np.expand_dims(x1, axis=0)
x2 = np.expand_dims(x2, axis=0)

# Make the predictions for the two images
y1 = model.predict(x1)
y2 = model.predict(x2)

# Get the predicted labels
label1 = labels[np.argmax(y1)]
label2 = labels[np.argmax(y2)]

print("Image 1: ", label1)
print("Image 2: ", label2)

# Compare the labels to determine the winner
if label1 == label2:
    print("It's a tie!")
elif label1 == 'rock':
    if label2 == 'paper':
        print("Image 2 wins with paper!")
    else:
        print("Image 1 wins with rock!")
elif label1 == 'paper':
    if label2 == 'scissors':
        print("Image 2 wins with scissors!")
    else:
        print("Image 1 wins with paper!")
else:
    if label2 == 'rock':
        print("Image 2 wins with rock!")
    else:
        print("Image 1 wins with scissors!")


# In[ ]:




