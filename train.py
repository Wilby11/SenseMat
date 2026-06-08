import torch
import os
from dataloader_windowfiltering import get_dataloaders
from models_guillermo.simple_cnn import SimpleCNN
from models_guillermo.cnn_lstm import SenseMat_CNN_LSTM
from models_guillermo.resnet_lstm import SenseMat_ResNet_LSTM
from models_guillermo.transformer import SenseMatTransformer
from models_guillermo.cnn_transformer import SenseMat_CNN_Transformer
from models_guillermo.trainer_pytorch import train_pytorch_model 

# ==========================================
# THE SWITCHBOARD
# ==========================================
SELECTED_MODEL = "SIMPLE-CNN" # Options: "SIMPLE-CNN", "CNN-LSTM", "RESNET-LSTM", "TRANSFORMER", "CNN-TRANSFORMER"
DATA_FOLDER = "" 

def main():
    print(f"=== SENSEMAT PIPELINE: {SELECTED_MODEL} ===")
    
    # 1. Load, Split, Window, and Batch (All in one line!)
    print("[1/2] Executing Data Pipeline (Loading, Windowing, Batching)...")
    
    # For SimpleCNN, we need the spatial grid, so flat_spatial=False
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=DATA_FOLDER,
        preprocessed="log",       # Switch between "log" and "non_log" easily
        window_size=20,           # Our agreed 0.5-second memory block
        flat_spatial=False,       # Keeps the 16x8 spatial grid
        use_metadata=False,       # Turn off age/weight features for now
        batch_size=32,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
        quality="standard",
        iqr_multiplier=3.0,
        debug=False
    )
    
    # 2. Training
    print(f"\n[2/2] Spinning up PyTorch Engine...")
    
    if SELECTED_MODEL == "SIMPLE-CNN":
        model = SimpleCNN()
    elif SELECTED_MODEL == "CNN-LSTM":
        model = SenseMat_CNN_LSTM()
    elif SELECTED_MODEL == "RESNET-LSTM":
        model = SenseMat_ResNet_LSTM()
    elif SELECTED_MODEL == "TRANSFORMER":
        model = SenseMatTransformer(
            window_size=20, 
            num_sensors=128, 
            d_model=128, 
            num_heads=4, 
            num_layers=1
        )
    elif SELECTED_MODEL == "CNN-TRANSFORMER":
        model = SenseMat_CNN_Transformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Define the path BEFORE training
    os.makedirs("saved_models", exist_ok=True)
    save_path = f"saved_models/{SELECTED_MODEL.lower()}_best.pth"

    # Run the custom training loop
    train_pytorch_model(model, train_loader, val_loader, optimizer, epochs=50, scheduler=scheduler, save_path=save_path)
    
    # Save the final weights
    os.makedirs("saved_models", exist_ok=True)
    save_path = f"saved_models/{SELECTED_MODEL.lower()}_v4.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n Training Complete. Weights saved to {save_path}")

    # Save the BEST weights
    best_path = f"saved_models/{SELECTED_MODEL.lower()}_best.pth"
    print(f"\n Training Complete. Best validation weights safely stored at {best_path}")

if __name__ == "__main__":
    main()