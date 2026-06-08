# 1. Import your model architectures
from models_guillermo.simple_cnn import SimpleCNN
from models_guillermo.cnn_lstm import SenseMat_CNN_LSTM
from models_guillermo.resnet_lstm import SenseMat_ResNet_LSTM
from models_guillermo.transformer import SenseMatTransformer
from models_guillermo.cnn_transformer import SenseMat_CNN_Transformer

# 2. Define the helper function
def count_parameters(model):
    """Counts only the trainable weights in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # 3. Instantiate the model (no weights or data needed)
    print("Spinning up architectures for physical sizing...")
    model = SenseMat_CNN_Transformer()
    
    # 4. Count the parameters
    total_params = count_parameters(model)
    
    # 5. Format for your Pareto Plot (in Millions)
    size_in_millions = total_params / 1_000_000
    
    print("\n=== MODEL SIZES ===")
    print(f"Model: {total_params:,} total weights")
    print(f"Plot Value: {size_in_millions:.2f}M")

if __name__ == "__main__":
    main()