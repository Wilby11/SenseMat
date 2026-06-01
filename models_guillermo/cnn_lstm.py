import torch
import torch.nn as nn

class SenseMat_CNN_LSTM(nn.Module):
    def __init__(self, lstm_hidden_size=128, lstm_layers=1):
        super().__init__()
        
        # 1. The Upgraded Spatial Feature Extractor (CNN V2)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), # Allows negative log-pressures to survive!
            nn.MaxPool2d(kernel_size=2), # Shrinks 16x8 to 8x4
            
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2), # Shrinks 8x4 to 4x2
            
            # Block 3 (No pooling here, we don't want to shrink it into oblivion)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            
            nn.Flatten() # Flattens 128 channels * 4 * 2 = 1024 features
        )
        
        # 2. Temporal Tracker (The LSTM)
        self.lstm = nn.LSTM(
            input_size=1024, # Updated to match the new CNN output
            hidden_size=lstm_hidden_size, 
            num_layers=lstm_layers, 
            batch_first=True
        )
        
        # 3. Final Output Head (6DoF)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        batch_size, time_steps, h, w = x.size()
        
        # Shape: [Batch * Time, 1, 16, 8]
        x_reshaped = x.view(batch_size * time_steps, 1, h, w)
        
        # Extract features
        cnn_features = self.cnn(x_reshaped) # Shape: [Batch * Time, 1024]
        
        # Shape: [Batch, Time, 1024]
        lstm_input = cnn_features.view(batch_size, time_steps, -1)
        
        # LSTM tracking
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # Grab final frame
        last_time_step = lstm_out[:, -1, :] 
        
        return self.fc(last_time_step)