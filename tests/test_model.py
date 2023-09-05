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
        if n == 100000000:
            break

    success_percentage = (correct_predictions / n) * 100

    with open('../results/scores_all.txt', 'a') as f:  # Open the file in append mode
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
    
    WEIGHTS_PATH = '../logs/all_classes/model_weights.h5'
    
    out_size = 6
    model = utils.model(out_size)
    
    # Compile the model
    model.compile(loss = "categorical_crossentropy", optimizer = Adam(learning_rate = 0.0001), metrics = ['accuracy'])
    
    # Load the saved weights
    model.load_weights(WEIGHTS_PATH)
    
    # Image for prediction
    image = Image.open(image_path).convert('L')                     # Convert image to grayscale
    image = image.resize((48, 48))                                  # Resize image to 48x48 pixels
    image_array = np.asarray(image).astype('float32') / 255.0       # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis = [0, -1])       # Add batch and color channel dimensions
    
    # Prediction
    predictions = model.predict(image_array)
    
    # Output
    predicted_class = np.argmax(predictions[0])

    if predicted_class == 0:
        return "angry"
    elif predicted_class == 1:
        return "fear"
    elif predicted_class == 2:
        return "happy"
    elif predicted_class == 3:
        return "neutral"
    elif predicted_class == 4:
        return "sad"
    else:
        return "surprise"

# This might need to change for the user
folders = ['/data/eurova/fer/test_tmp/surprise', '/data/eurova/fer/test_tmp/angry', '/data/eurova/fer/test_tmp/happy', 
           '/data/eurova/fer/test_tmp/neutral', '/data/eurova/fer/test_tmp/fear', '/data/eurova/fer/test_tmp/sad']

open('../results/scores_all.txt', 'w').close()

for folder in folders:
    test_folder(folder)
