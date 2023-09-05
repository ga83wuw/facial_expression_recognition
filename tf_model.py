import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import csv

import tf_utils as utils

train_dir = "/data/eurova/fer/train_neg"
test_dir = "/data/eurova/fer/test_neg"

emotion_labels = [subfolder for subfolder in os.listdir(train_dir) if os.path.isdir(train_dir)]
print(emotion_labels)
out_size = len(emotion_labels)
epochs = 100

train_generator, validation_generator = utils.dataloader(train_dir, test_dir)

model = utils.model(out_size)
# Compile the model with categorical cross-entropy loss, adam optimizer, and accuracy metric
model.compile(loss = "categorical_crossentropy", optimizer = tf.keras.optimizers.Adam(lr = 0.0001), metrics = ['accuracy'])

# Define the callback
checkpoint_callback = ModelCheckpoint(
    filepath = 'model_weights.h5',
    monitor = 'val_accuracy',
    save_best_only = True,
    save_weights_only = True,
    mode = 'max',
    verbose = 1
)


# Train the model with the callback
history = model.fit(
    train_generator,
    steps_per_epoch = len(train_generator),
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = len(validation_generator),
    callbacks = [checkpoint_callback]
)

# Define CSV file and write the header
with open('metrics.csv', 'w', newline = '') as csvfile:
    fieldnames = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Loop through epochs and write metrics
    for epoch in range(len(history.history['loss'])):
        writer.writerow({
            'epoch': epoch + 1,                                     # epoch index starts from 1
            'train_loss': history.history['loss'][epoch],
            'train_accuracy': history.history['accuracy'][epoch],
            'val_loss': history.history['val_loss'][epoch],
            'val_accuracy': history.history['val_accuracy'][epoch]
        })
