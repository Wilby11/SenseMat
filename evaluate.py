import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader import get_dataloaders
from models_guillermo.simple_cnn import SimpleCNN
from models_guillermo.cnn_lstm import SenseMat_CNN_LSTM
from models_guillermo.resnet_lstm import SenseMat_ResNet_LSTM
from models_guillermo.transformer import SenseMatTransformer
from models_guillermo.cnn_transformer import SenseMat_CNN_Transformer

# ==========================================
# CONFIGURATION
# ==========================================
SELECTED_MODEL = "CNN-LSTM" # Options: "SIMPLE-CNN", "CNN-LSTM", "RESNET-LSTM", "TRANSFORMER", "CNN-TRANSFORMER"
MODEL_WEIGHTS = "saved_models/cnn-lstm_v1.pth"
DATA_ROOT = ""

def plot_tracking_results(targets, predictions):
    """Generates a visual comparison of True vs Predicted trajectories."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('TrackIR vs Neural Network Predictions (Unseen Test Data)', fontsize=16)
    
    titles = ['X Position', 'Y Position', 'Z Position', 'Pitch', 'Yaw', 'Roll']
    
    for i, ax in enumerate(axes.flatten()):
        # Plot only the first 200 windows so the graph isn't a solid block of ink
        ax.plot(targets[:200, i], label='True TrackIR', color='blue', linewidth=2)
        ax.plot(predictions[:200, i], label='Model Prediction', color='orange', linestyle='dashed', linewidth=2)
        ax.set_title(titles[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("evaluation_plot.png")
    print("\n Saved visual trajectory plot to 'evaluation_plot.png'")

def main():
    print(f"=== SENSEMAT EVALUATION: {SELECTED_MODEL} ===")
    
    # 1. Load the Data (We only care about the test_loader)
    print("[1/3] Extracting unseen Test Data...")
    _, _, test_loader = get_dataloaders(
        data_root=DATA_ROOT,
        preprocessed="log", # Switch between "log" and "non_log" easily
        window_size=20,
        flat_spatial=False,
        use_metadata=False,
        batch_size=32,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42
    )
    
    # 2. Initialize Model and Load Weights
    print("[2/3] Loading trained PyTorch weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # map_location ensures it works even if you trained on GPU but are evaluating on CPU
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    model.eval() # CRITICAL: Turns off dropout and batch norm variations
    
    # 3. Inference Loop
    print("[3/3] Running Inference...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad(): # Saves memory, turns off gradient tracking
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            
            # Predict the 6DoF
            preds = model(batch_x)
            
            # SLICE: We only want the True TrackIR label from the final frame of the window
            targets = batch_y[:, -1, :].numpy() 
            
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets)
            
    # Stack lists into final massive arrays
    predictions_array = np.vstack(all_predictions)
    targets_array = np.vstack(all_targets)
    
    # 4. Calculate Clinical Metrics (Mean Absolute Error)
    # MAE is much easier for humans to interpret than Mean Squared Error
    mae_per_axis = np.mean(np.abs(predictions_array - targets_array), axis=0)
    
    print("\n=== CLINICAL ACCURACY (Mean Absolute Error) ===")
    print(f"Position X: {mae_per_axis[0]:.4f}")
    print(f"Position Y: {mae_per_axis[1]:.4f}")
    print(f"Position Z: {mae_per_axis[2]:.4f}")
    print("---")
    print(f"Rotation Pitch: {mae_per_axis[3]:.4f}")
    print(f"Rotation Yaw  : {mae_per_axis[4]:.4f}")
    print(f"Rotation Roll : {mae_per_axis[5]:.4f}")
    print("===============================================")
    
    # 5. Generate Visuals
    plot_tracking_results(targets_array, predictions_array)

if __name__ == "__main__":
    main()