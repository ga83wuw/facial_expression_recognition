import os
import shutil
import random

# Path to the main folder containing all class folders
main_folder_path = "/data/eurova/fer/train_balanced"

# Minimum number of images required in each class
min_images = 6000

for class_folder in os.listdir(main_folder_path):
    class_folder_path = os.path.join(main_folder_path, class_folder)
    
    # Ensure it's a folder
    if not os.path.isdir(class_folder_path):
        continue

    image_files = [f for f in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, f))]
    num_images = len(image_files)
    
    if num_images < min_images:
        num_required = min_images - num_images
        print(f"Class {class_folder} has {num_images} images, duplicating {num_required} images.")
        
        for i in range(num_required):
            # Randomly select an image to duplicate
            image_to_copy = random.choice(image_files)
            original_image_path = os.path.join(class_folder_path, image_to_copy)
            
            # Create a new name for the duplicated image
            new_image_name = f"duplicated_{i}_" + image_to_copy
            new_image_path = os.path.join(class_folder_path, new_image_name)
            
            # Copy the image
            shutil.copy2(original_image_path, new_image_path)