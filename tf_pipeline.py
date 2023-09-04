import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from PIL import Image

import tf_utils as utils

WEIGHTS_PATH_NEG = './pipeline/...'
WEIGHTS_PATH_POS = './pipeline/...'
WEIGHTS_PATH_POSNEG = './pipeline/...'

out_size = 2
model = utils.model(out_size)

# Compile the model
model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = 0.0001), metrics = ['accuracy'])

# Load the saved weights
model.load_weights(WEIGHTS_PATH_POSNEG)

# Image for prediction
image_path = 'path/to/your/image.jpg'
image = Image.open(image_path).convert('L')                     # Convert image to grayscale
image = image.resize((48, 48))                                  # Resize image to 48x48 pixels
image_array = np.asarray(image).astype('float32') / 255.0       # Normalize to [0, 1]
image_array = np.expand_dims(image_array, axis = [0, -1])       # Add batch and color channel dimensions

# Prediction
predictions = model.predict(image_array)

# Output
predicted_class = np.argmax(predictions[0])
print(f"Predicted class: {predicted_class}")