import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# Define transforms for data preprocessing and augmentation

transform = transforms.Compose([
    transforms.ToTensor()
    #transforms.Normalize(mean = [0.5], std = [0.5])  # Normalize to [0, 1]
])

# Define a custom dataset class
class FacialDataset(Dataset):
    def __init__(self, root_dir, labels, transform = transform):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = labels
        self.image_paths = []
        self.label_indices = []

        for label_idx, emotion in enumerate(labels):
            emotion_path = os.path.join(self.root_dir, 'train', emotion)
            image_files = os.listdir(emotion_path)
            
            for image_file in image_files:
                image_path = os.path.join(emotion_path, image_file)
                self.image_paths.append(image_path)
                self.label_indices.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_idx = self.label_indices[idx]
        
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        label = label_idx
        
        if self.transform:
            image = self.transform(image)

        image = np.array(image, dtype = 'float32') / 255.
        image = torch.from_numpy(image).float()
        
        return image, label