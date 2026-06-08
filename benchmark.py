import torch
import time
import numpy as np

# Import your champion model
from models_guillermo.simple_cnn import SimpleCNN
from models_guillermo.cnn_lstm import SenseMat_CNN_LSTM
from models_guillermo.resnet_lstm import SenseMat_ResNet_LSTM
from models_guillermo.transformer import SenseMatTransformer
from models_guillermo.cnn_transformer import SenseMat_CNN_Transformer

def main():
    print("=== SENSEMAT LATENCY BENCHMARK ===")
    
    # 1. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Hardware: {device.type.upper()}")
    
    # 2. Load Model
    model = SenseMat_CNN_Transformer()
    model.load_state_dict(torch.load("saved_models/cnn-transformer_v4.pth", map_location=device))
    model.to(device)
    model.eval()  # CRITICAL: Turns off Dropout and BatchNorm layers
    
    # 3. Create a "Dummy" Hospital Bed Window
    # Shape: (Batch=1, Time=20, Rows=16, Cols=8)
    dummy_input = torch.randn(1, 20, 16, 8, dtype=torch.float32).to(device)
    
    # ==========================================
    # PHASE 1: THE WARM-UP
    # ==========================================
    # PyTorch is slow on its first few inferences because it has to allocate RAM/VRAM.
    print("\nWarming up silicon...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
            
    # ==========================================
    # PHASE 2: THE SPRINT
    # ==========================================
    ITERATIONS = 1000
    times = []
    
    print(f"Running {ITERATIONS} continuous inferences...")
    
    with torch.no_grad():
        for _ in range(ITERATIONS):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            
            # If using GPU, we must force PyTorch to wait until the calculation is actually finished
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert seconds to milliseconds

    # ==========================================
    # 4. DIAGNOSIS
    # ==========================================
    avg_time = np.mean(times)
    p95_time = np.percentile(times, 95)  # 95% of predictions are faster than this
    
    print("\n=== RESULTS ===")
    print(f"Average Latency: {avg_time:.2f} ms")
    print(f"95th Percentile: {p95_time:.2f} ms")
    
    # The ultimate deployment check
    if avg_time < 500.0:
        print("\n✅ DEPLOYMENT STATUS: PASS")
        print("Model processes a 0.5s window faster than real-time.")
    else:
        print("\n❌ DEPLOYMENT STATUS: FAIL")
        print("Model takes longer to think than the movement takes to happen.")

if __name__ == "__main__":
    main()