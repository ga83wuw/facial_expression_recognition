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
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

from dataloader import FacialDataset
from utils import EarlyStopper
from models import FERModel
from models import FERModel_simple
from models import MyVGGModel
from models import vgg13, vgg16

### DATA ###

# Define the path to your dataset folder
dataset_path = '/data/eurova/fer/'
train_folder_path = os.path.join(dataset_path, 'train')

# Define emotion labels and other variables
emotion_labels = [subfolder for subfolder in os.listdir(train_folder_path) if os.path.isdir(os.path.join(train_folder_path, subfolder))]
print(emotion_labels)
num_classes = len(emotion_labels)

DISPLAY = False

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

### ALTERNATIVE DATA APPROACH ###
from torchvision.datasets import ImageFolder

# Directory paths
train_dir = "/data/eurova/fer/train"
test_dir = "/data/eurova/fer/test"

# Data transforms
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5], std = [0.5]) 
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5], std = [0.5])
])

# Datasets
train_dataset = ImageFolder(root = train_dir, transform = train_transform)
test_dataset = ImageFolder(root = test_dir, transform = test_transform)


# Split the data into train and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Data loaders
batch_size = 64
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False)

### --- ###

### MODEL ###

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
print(device)

# Instantiate the model
num_classes = len(emotion_labels)

#model = FERModel(num_classes = num_classes).to(device)
#model = FERModel_simple().to(device)
#model = MyVGGModel(num_classes = num_classes)
#model = vgg13(pretrained = True)
model = vgg16(pretrained = True)

initial_layer = nn.Conv2d(1, 64, kernel_size = 3, padding = 1)
model.features[0] = initial_layer
print(model.features)
#print(*list(model.features.children())[0:])

num_ftrs = model.classifier[3].out_features

model.classifier = nn.Sequential(nn.Linear(7 * 7 * 512, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(1024, num_classes))

# Iterate through the layers and freeze except for the specified ones
# for idx, layer in enumerate(model.features):
#     if idx == 0 or idx >= len(model.features) - 3:
#         # Set requires_grad to True for the first layer and last 3 layers
#         for param in layer.parameters():
#             param.requires_grad = True
#     else:
#         # Freeze all other layers
#         for param in layer.parameters():
#             param.requires_grad = False
model.eval()

model.to(device)

# Define hyperparameters
num_epochs = 50
learning_rate = 0.001

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)

metrics = []

# Initialize the EarlyStopping callback
patience = 5
min_delta = 0
early_stopper = EarlyStopper(patience = patience, min_delta = min_delta)

# Initialize variables to keep track of the best model
best_model_state_dict = None
best_val_loss = np.inf

# Training loop
for epoch in range(num_epochs):
    model.train()               # Set the model to training mode

    running_loss = 0.0
    true_positives = 0          # Count of true positive predictions
    predicted_positives = 0     # Count of all predicted positives
    actual_positives = 0        # Count of all actual positives
    correct_predictions = 0     # Count of correct predictions

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        _, predicted_train = torch.max(outputs.data, 1)

        # Accuracy
        correct_predictions += (predicted_train == labels).sum().item()
        
        # Precision & Recall & F1
        true_positives += ((predicted_train == 1) & (labels == 1)).sum().item()     # Count true positives
        predicted_positives += (predicted_train == 1).sum().item()                  # Count all predicted positives
        actual_positives += (labels == 1).sum().item()                              # Count all actual positives
        
    
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / len(train_loader.dataset)
    train_precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    train_recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0.0
    
    # Validation loop
    model.eval()                # Set the model to evaluation mode
    
    val_loss = 0.0
    correct_predictions = 0     # Count of correct predictions
    true_positives = 0          # Count of true positive predictions
    predicted_positives = 0     # Count of all predicted positives
    actual_positives = 0        # Count of all actual positives

    b_val_recall = 0.0
    b_val_f1 = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)

            loss_val = criterion(outputs, labels)
            val_loss += loss_val.item()
            
            _, predicted_val = torch.max(outputs.data, 1)

            # Accuracy
            correct_predictions += (predicted_val == labels).sum().item()
            
            # Precision & Recall & F1
            true_positives += ((predicted_val == 1) & (labels == 1)).sum().item()   # Count true positives            
            predicted_positives += (predicted_val == 1).sum().item()                # Count all predicted positives
            actual_positives += (labels == 1).sum().item()                          # Count all actual positives
    
    # Loss
    avg_val_loss = val_loss / len(val_loader)
    # Accuracy
    val_accuracy = correct_predictions / len(val_loader.dataset)
    # Precision
    val_precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    # Recall
    val_recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    # F1-score
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0.0

    # Log metrics to the metrics list
    metrics.append({'epoch': epoch + 1, 
                    'train_loss': avg_train_loss, 
                    'train_acc': train_accuracy, 
                    'train_pre': train_precision, 
                    'train_rec': train_recall, 
                    'train_f1': train_f1, 
                    'val_loss': avg_val_loss, 
                    'val_acc': val_accuracy,
                    'val_pre': val_precision,
                    'val_rec': val_recall,
                    'val_f1': val_f1})
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Train Accuracy: {train_accuracy:.2%}, "
          f"Train Precision: {train_precision:.2%}, "
          f"Train Recall: {train_recall:.2%}, "
          f"Train F1: {train_f1:.2%}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.2%}, "
          f"Val Precision: {val_precision:.2%}, "
          f"Val Recall: {val_recall:.2%}, "
          f"Val F1: {val_f1:.2%} .-")
    
    # Save model's state dictionary if validation loss improved
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state_dict = model.state_dict()

    # Check if early stopping should be triggered
    if early_stopper.early_stop(avg_val_loss):             
        break

# Logs directory
if not os.path.exists('./logs'):
    # Create the folder
    os.makedirs(logs_folder)
    print(f"Logs directory created.")
else:
    print(f"Logs directory already exists.")

# Save metrics to a CSV file
csv_file = './logs/training_metrics.csv'
with open(csv_file, mode = 'w', newline = '') as file:
    writer = csv.DictWriter(file, fieldnames = ['epoch', 
                                                'train_loss', 'train_acc', 'train_pre', 'train_rec', 'train_f1', 
                                                'val_loss', 'val_acc', 'val_pre', 'val_rec', 'val_f1'])
    writer.writeheader()
    writer.writerows(metrics)

# Save the best model's state dictionary
pth_path = './logs/best_model.pth'
if best_model_state_dict is not None:
    torch.save(best_model_state_dict, pth_path)

print("Training finished.")