import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.utils import make_grid

from dataloader import FacialDataset
from models import FERModel

### DATA ###

# Define the path to your dataset folder
dataset_path = './archive/'

# Define emotion labels and other variables
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_classes = len(emotion_labels)

DISPLAY = False

# Create datasets and data loaders
facial_dataset = FacialDataset(dataset_path, emotion_labels)
train_size = int(0.9 * len(facial_dataset))
val_size = len(facial_dataset) - train_size
train_dataset, val_dataset = random_split(facial_dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

if DISPLAY:
    # Display a batch of augmented images
    def show_batch(images, labels):
        num_images = images.shape[0]
        num_rows = int(np.ceil(num_images / 8))
        
        fig, axes = plt.subplots(num_rows, 8, figsize = (20, 3 * num_rows))
        for i, ax in enumerate(axes.flatten()):
            if i < num_images:
                ax.imshow(np.transpose(images[i], (1, 2, 0)), cmap = 'gray')
                ax.set_title(emotion_labels[labels[i]])
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    # Visualize a batch of augmented training images
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    show_batch(images, labels)

### MODEL ###

# Instantiate the FERModel
num_classes = len(emotion_labels)
fer_model = FERModel(num_classes)

# Define hyperparameters
num_epochs = 10
learning_rate = 0.001

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fer_model.parameters(), lr = learning_rate)

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
print(device)

# Instantiate the model
model = FERModel(num_classes = num_classes).to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    val_accuracy = correct_predictions / total_predictions
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.2%}")

print("Training finished.")