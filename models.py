import torch.nn as nn

INIT_CHANNELS = 32

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        return self.conv(x)

class FERModel(nn.Module):
    def __init__(self, num_classes):
        super(FERModel, self).__init__()
        self.conv1 = ConvBlock(in_channels = 1, out_channels = INIT_CHANNELS)
        self.conv2 = ConvBlock(in_channels = INIT_CHANNELS, out_channels = 2 * INIT_CHANNELS)
        self.conv3 = ConvBlock(in_channels = 2 * INIT_CHANNELS, out_channels = 4 * INIT_CHANNELS)
        self.fc1 = nn.Linear(4 * INIT_CHANNELS * 6 * 6, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.fc1(x)
        x = self.dropout(x)
        x = nn.functional.softmax(x, dim = 1)
        x = self.fc2(x)
        
        return x