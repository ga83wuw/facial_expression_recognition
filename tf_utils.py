import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

def model(out_size = 2):

    # Define the model architecture
    model = Sequential()

    # Add a convolutional layer with 32 filters, 3x3 kernel size, and relu activation function
    model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (48, 48, 1)))
    # Add a batch normalization layer
    model.add(BatchNormalization())
    # Add a second convolutional layer with 64 filters, 3x3 kernel size, and relu activation function
    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    # Add a second batch normalization layer
    model.add(BatchNormalization())
    # Add a max pooling layer with 2x2 pool size
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Add a dropout layer with 0.25 dropout rate
    model.add(Dropout(0.25))

    # Add a third convolutional layer with 128 filters, 3x3 kernel size, and relu activation function
    model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
    # Add a third batch normalization layer
    model.add(BatchNormalization())
    # Add a fourth convolutional layer with 128 filters, 3x3 kernel size, and relu activation function
    model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
    # Add a fourth batch normalization layer
    model.add(BatchNormalization())
    # Add a max pooling layer with 2x2 pool size
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Add a dropout layer with 0.25 dropout rate
    model.add(Dropout(0.25))

    # Add a fifth convolutional layer with 256 filters, 3x3 kernel size, and relu activation function
    model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu'))
    # Add a fifth batch normalization layer
    model.add(BatchNormalization())
    # Add a sixth convolutional layer with 256 filters, 3x3 kernel size, and relu activation function
    model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu'))
    # Add a sixth batch normalization layer
    model.add(BatchNormalization())
    # Add a max pooling layer with 2x2 pool size
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Add a dropout layer with 0.25 dropout rate
    model.add(Dropout(0.25))

    # Flatten the output of the convolutional layers
    model.add(Flatten())
    # Add a dense layer with 256 neurons and relu activation function
    model.add(Dense(256, activation = 'relu'))
    # Add a seventh batch normalization layer
    model.add(BatchNormalization())
    # Add a dropout layer with 0.5 dropout rate
    model.add(Dropout(0.5))
    # Add a dense layer with 7 neurons (one for each class) and softmax activation function
    model.add(Dense(out_size, activation = 'softmax'))

    return model

def dataloader(train_dir = '', test_dir = ''):

    train_datagen = ImageDataGenerator(
        width_shift_range = 0.1,        # Randomly shift the width of images by up to 10%
        height_shift_range = 0.1,       # Randomly shift the height of images by up to 10%
        horizontal_flip = True,         # Flip images horizontally at random
        rescale = 1./255,               # Rescale pixel values to be between 0 and 1
        validation_split = 0.2          # Set aside 20% of the data for validation
    )

    validation_datagen = ImageDataGenerator(
        rescale = 1./255,               # Rescale pixel values to be between 0 and 1
        validation_split = 0.2          # Set aside 20% of the data for validation
    )

    train_generator = train_datagen.flow_from_directory(
        directory = train_dir,           # Directory containing the training data
        target_size = (48, 48),          # Resizes all images to 48x48 pixels
        batch_size = 64,                 # Number of images per batch
        color_mode = "grayscale",        # Converts the images to grayscale
        class_mode = "categorical",      # Classifies the images into 7 categories
        subset = "training"              # Uses the training subset of the data
    )

    test_generator = validation_datagen.flow_from_directory(
        directory = test_dir,            # Directory containing the validation data
        target_size = (48, 48),          # Resizes all images to 48x48 pixels
        batch_size = 64,                 # Number of images per batch
        color_mode = "grayscale",        # Converts the images to grayscale
        class_mode = "categorical",      # Classifies the images into 7 categories
        subset = "validation"            # Uses the validation subset of the data
    )

    return train_generator, test_generator