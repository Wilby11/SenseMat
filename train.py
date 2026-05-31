import torch
import os
from dataloader import get_dataloaders
from models_guillermo.simple_cnn import SimpleCNN
from models_guillermo.trainer_pytorch import train_pytorch_model 

# ==========================================
# THE SWITCHBOARD
# ==========================================
SELECTED_MODEL = "SIMPLE-CNN"
# Point this to either the log or non-log CSV
DATA_FOLDER = "" 

def main():
    print(f"=== SENSEMAT PIPELINE: {SELECTED_MODEL} ===")
    
    # 1. Load, Split, Window, and Batch (All in one line!)
    print("[1/2] Executing Data Pipeline (Loading, Windowing, Batching)...")
    
    # For SimpleCNN, we need the spatial grid, so flat_spatial=False
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=DATA_FOLDER,
        preprocessed="non_log",       # Switch between "log" and "non_log" easily
        window_size=20,           # Our agreed 0.5-second memory block
        flat_spatial=False,       # Keeps the 16x8 spatial grid
        use_metadata=False,       # Turn off age/weight features for now
        batch_size=32,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42
    )
    
    # 2. Training
    print(f"\n[2/2] Spinning up PyTorch Engine...")
    
    if SELECTED_MODEL == "SIMPLE-CNN":
        model = SimpleCNN()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Run the custom training loop
    train_pytorch_model(model, train_loader, val_loader, optimizer, epochs=50)
    
    # Save the final weights
    os.makedirs("saved_models", exist_ok=True)
    save_path = f"saved_models/{SELECTED_MODEL.lower()}_v1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Training Complete. Weights saved to {save_path}")

if __name__ == "__main__":
    main()