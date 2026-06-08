import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, window_size=20): # Make sure this matches your dataloader!
        super().__init__()
        
        # 1. The Upgraded Spatial Feature Extractor (CNN V2)
        # Instead of 1 channel per frame, we ingest the whole time window simultaneously
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=window_size, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), 
            nn.MaxPool2d(kernel_size=2), # Shrinks 16x8 to 8x4
            
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2), # Shrinks 8x4 to 4x2
            
            # Block 3 (No pooling, mirroring the CNN-LSTM)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            
            nn.Flatten() # Flattens 128 channels * 4 * 2 = 1024 features
        )
        
        # 2. Final Output Head (6DoF)
        # We replace the LSTM with a deeper, robust fully-connected block
        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        # Input shape from dataloader: [Batch, Time, 16_Rows, 8_Cols]
        # PyTorch Conv2d expects: [Batch, Channels, Height, Width]
        # Because we want Time to act as our Channels, the dataloader shape is already perfect!
        
        features = self.cnn(x)     # Extracts [Batch, 1024]
        out = self.fc(features)    # Maps to [Batch, 6]
        
        return out