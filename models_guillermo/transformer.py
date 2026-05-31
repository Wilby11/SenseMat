import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Transformers have no concept of sequential time. 
    We must mathematically tag each frame with its position (0 to 19).
    """
    def __init__(self, d_model, max_len=20):
        super().__init__()
        # Create a learnable embedding layer for the timeline
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x shape: (Batch, seq_len, Features)
        seq_len = x.size(1)
        # Create an array [0, 1, 2... 19]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        # Expand it to match the batch size
        positions = positions.unsqueeze(0).expand(x.size(0), -1)
        # Add the positional tag to the raw sensor data
        return x + self.pos_embedding(positions)


class SenseMatTransformer(nn.Module):
    def __init__(self, window_size=20, num_sensors=128, d_model=128, num_heads=4, num_layers=1):
        super().__init__()
        
        # 1. Input Projection
        # Ensures our 128 sensors match the Transformer's expected dimension size
        self.input_projection = nn.Linear(num_sensors, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=window_size)
        
        # 3. The Transformer Block
        # We define a single layer, then wrap it in the Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=256, 
            dropout=0.1, 
            batch_first=True  # Crucial: Forces [Batch, Time, Features] format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Head (Regression to 6DoF)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 6) # Final 6 linear nodes
        )

    def forward(self, x):
        # Incoming x shape: (Batch, 20, 128)
        
        # Project and add timestamps
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Pass through the Multi-Head Attention mechanism
        x = self.transformer_encoder(x)
        
        # Collapse the time dimension (Global Average Pooling)
        # x is currently (Batch, 20, 128). We average across the 20 frames.
        x = torch.mean(x, dim=1)
        
        # Predict the 6DoF target
        out = self.output_head(x)
        return out