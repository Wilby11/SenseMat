import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """A custom mini-residual block that doesn't shrink the grid."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Linear activation (no ReLU) on the second conv before the add
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += shortcut # The skip connection!
        return self.relu(out)

class SenseMat_ResNet_LSTM(nn.Module):
    def __init__(self, lstm_hidden_size=64, lstm_layers=1):
        super().__init__()
        
        # 1. Build the isolated Spatial Engine (Mini-ResNet)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.res1 = ResBlock(32)
        self.res2 = ResBlock(32)
        
        # Flattening 32 channels * 16 rows * 8 cols = 4096 features per frame
        self.flatten = nn.Flatten() 
        
        # 2. The Temporal Tracker (LSTM)
        self.lstm = nn.LSTM(
            input_size=4096, 
            hidden_size=lstm_hidden_size, 
            num_layers=lstm_layers, 
            batch_first=True
        )
        
        # 3. Output Layer (6 Linear Nodes for 6DoF)
        self.fc = nn.Linear(lstm_hidden_size, 6)

    def forward(self, x):
        # Incoming 'x' shape from PyTorch dataloader: [Batch, Time, 16, 8]
        batch_size, time_steps, h, w = x.size()
        
        # Reshape to treat every frame as an independent image for the CNN
        # New shape: [Batch * Time, 1 (Channel), 16, 8]
        x_reshaped = x.view(batch_size * time_steps, 1, h, w)
        
        # Extract features using the ResNet Engine
        out = self.initial_conv(x_reshaped)
        out = self.res1(out)
        out = self.res2(out)
        spatial_features = self.flatten(out) # Shape: [Batch * Time, 4096]
        
        # Reshape back into a chronological sequence for the LSTM
        # New shape: [Batch, Time, 4096]
        lstm_input = spatial_features.view(batch_size, time_steps, -1)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # We only care about the LSTM's conclusion at the VERY LAST frame
        last_time_step = lstm_out[:, -1, :] # Shape: [Batch, 64]
        
        # Predict the 6DoF
        return self.fc(last_time_step)