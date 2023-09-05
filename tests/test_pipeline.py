import os

def test_folder(folder_path):
    label_from_folder = os.path.basename(folder_path)  
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    correct_predictions = 0
    n = 0
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        predicted_class = test_pipeline(image_path)  

        if predicted_class == label_from_folder:
            correct_predictions += 1
        
        n += 1
        if n == 10000000000:
            break

    success_percentage = (correct_predictions / n) * 100

    with open('../results/scores_step.txt', 'a') as f:  # Open the file in append mode
        f.write(f"Accuracy for folder {folder_path}: {success_percentage:.2f}%\n")


    print(f"Accuracy for folder {folder_path}: {success_percentage:.2f}%")
    return success_percentage

# Modify test_pipeline to return the predicted class

def test_pipeline(image_path):
    
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from PIL import Image
    import matplotlib.pyplot as plt
    
    import tf_utils as utils
    
    WEIGHTS_PATH_NEG = '../logs/neg/model_weights.h5'
    WEIGHTS_PATH_POS = '../logs/pos/model_weights.h5'
    WEIGHTS_PATH_POSNEG = '../logs/posneg/model_weights.h5'
    
    out_size = 2
    model = utils.model(out_size)
    
    # Compile the model
    model.compile(loss = "categorical_crossentropy", optimizer = Adam(learning_rate = 0.0001), metrics = ['accuracy'])
    
    # Load the saved weights
    model.load_weights(WEIGHTS_PATH_POSNEG)
    
    # Image for prediction
    image = Image.open(image_path).convert('L')                     # Convert image to grayscale
    image = image.resize((48, 48))                                  # Resize image to 48x48 pixels
    image_array = np.asarray(image).astype('float32') / 255.0       # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis = [0, -1])       # Add batch and color channel dimensions
    
    #print("Image has been loaded successfully.")
        
    #print("Predictions on the way.")
    
    # Prediction
    predictions = model.predict(image_array)
    
    # Output
    predicted_class = np.argmax(predictions[0])

    if predicted_class == 0:
        
        out_size = 3
        model = utils.model(out_size)
        
        # Compile the model
        model.compile(loss = "categorical_crossentropy", optimizer = Adam(learning_rate = 0.0001), metrics = ['accuracy'])
        
        # Load the saved weights
        model.load_weights(WEIGHTS_PATH_NEG)
        # Prediction
        predictions = model.predict(image_array)
        # Output
        predicted_class = np.argmax(predictions[0])

        if predicted_class == 0:
            return "angry"
        elif predicted_class == 1:
            return "fear"
        else:
            return "sad"
    else:
        
        out_size = 3
        model = utils.model(out_size)
        
        # Compile the model
        model.compile(loss = "categorical_crossentropy", optimizer = Adam(learning_rate = 0.0001), metrics = ['accuracy'])
        
        # Load the saved weights
        model.load_weights(WEIGHTS_PATH_POS)
        # Prediction
        predictions = model.predict(image_array)
        # Output
        predicted_class = np.argmax(predictions[0])

        if predicted_class == 0:
            return "happy"
        elif predicted_class == 1:
            return "neutral"
        else:
            return "surprise"

# This might need to change for the user
folders = ['/data/eurova/fer/test/surprise', '/data/eurova/fer/test/angry', '/data/eurova/fer/test/happy', 
           '/data/eurova/fer/test/neutral', '/data/eurova/fer/test/fear', '/data/eurova/fer/test/sad']

open('../results/scores_step.txt', 'w').close()

for folder in folders:
    test_folder(folder)
