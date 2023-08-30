import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# Define transforms for data preprocessing and augmentation

transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL image
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5], std = [0.5])  # Normalize to [-1, 1]
])

# Define a custom dataset class
class FacialDataset(Dataset):
    def __init__(self, root_dir, labels, transform = transform):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = labels
        self.image_paths = []
        self.labels = []

        for label_idx, emotion in enumerate(labels):
            emotion_path = os.path.join(self.root_dir, 'train', emotion)
            image_files = os.listdir(emotion_path)
            
            for image_file in image_files:
                image_path = os.path.join(emotion_path, image_file)
                image = plt.imread(image_path)

                # Check if the image is already grayscale
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis = -1)  # Add channel dimension
                self.image_paths.append(image_path)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = plt.imread(image_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label