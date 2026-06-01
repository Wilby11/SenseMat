import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Tags each frame mathematically so the Transformer knows the chronological order."""
    def __init__(self, d_model, max_len=20):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(x.size(0), -1)
        return x + self.pos_embedding(positions)

class SenseMat_CNN_Transformer(nn.Module):
    def __init__(self, window_size=20, d_model=256, num_heads=4, num_layers=1):
        super().__init__()
        
        # 1. The Spatial Feature Extractor (CNN V2)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2), # Shrinks 16x8 to 8x4
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2), # Shrinks 8x4 to 4x2
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Flatten() # Flattens to 128 * 4 * 2 = 1024 features per frame
        )
        
        # 2. The Bridge
        # Compresses the heavy CNN output into a sleek vector for the Transformer
        self.feature_projection = nn.Sequential(
            nn.Linear(1024, d_model),
            nn.LeakyReLU(0.1)
        )
        
        # 3. The Temporal Tracker (Transformer)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=window_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model * 2, 
            dropout=0.2, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Output Head (6DoF)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        batch_size, time_steps, h, w = x.size()
        
        # --- SPATIAL EXTRACTION ---
        x_reshaped = x.view(batch_size * time_steps, 1, h, w)
        cnn_features = self.cnn(x_reshaped) 
        
        # Bridge to Transformer dimension
        projected_features = self.feature_projection(cnn_features)
        
        # --- TEMPORAL TRACKING ---
        # Reshape back to chronological timeline: [Batch, Time, Features]
        transformer_input = projected_features.view(batch_size, time_steps, -1)
        
        # Add timestamps and calculate Attention
        transformer_input = self.pos_encoder(transformer_input)
        transformer_out = self.transformer_encoder(transformer_input)
        
        # Collapse the 20-frame timeline by averaging the Attention weights
        global_context = torch.mean(transformer_out, dim=1)
        
        return self.fc(global_context)