import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # PyTorch Conv2d expects inputs in shape: [Batch, Channels, Height, Width]
        # Our shape will be: [Batch, 20_frames, 8_rows, 16_cols]
        
        # 1. Spatial Feature Extractor
        self.conv1 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks 8x16 to 4x8
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks 4x8 to 2x4
        
        self.flatten = nn.Flatten()
        
        # 2. Output Head
        # 64 channels * 2 rows * 4 cols = 512 flat features
        self.fc1 = nn.Linear(512, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5) # Randomly turns off 50% of the neurons during training
        self.fc2 = nn.Linear(128, 6) # 6DoF Output

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x) # Apply dropout before the final decision
        out = self.fc2(x)
        return out