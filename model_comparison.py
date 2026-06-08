import matplotlib.pyplot as plt
import numpy as np

# --- 1. YOUR HARD DATA ---
models =          ['Simple CNN', 'CNN-LSTM', 'ResNet-LSTM', 'Transformer', 'CNN-Trans']
mae_scores =      [1.98,         1.96,       2.06,          2.02,          2.05] # weighted mean: (x+y+z/3)+(roll+pitch+yaw/30)
inference_times = [0.65,         2.77,       4.05,          0.85,          3.18] # in milliseconds
model_sizes =     [0.24,         0.69,       1.10,          0.16,          0.90]  # in Millions of Parameters

def plot_pareto_frontiers():
    print("=== GENERATING PARETO FRONTIER PLOTS ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('SenseMat real-time implementation: Accuracy vs. Speed and Size', fontsize=16, fontweight='bold')
    
    # --- Plot 1: MAE vs Inference Time ---
    ax1 = axes[0]
    ax1.scatter(inference_times, mae_scores, s=150, c='royalblue', edgecolors='black', zorder=3)
    
    for i, txt in enumerate(models):
        ax1.annotate(txt, (inference_times[i], mae_scores[i]), 
                     xytext=(10, 5), textcoords='offset points', fontsize=10)
        
    ax1.set_title('Real-Time Latency & Hardware Efficiency', fontsize=14)
    ax1.set_xlabel('Inference latency (Milliseconds)', fontsize=12)
    ax1.set_ylabel('Weighted Mean Absolute Error', fontsize=12)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.invert_yaxis() # Invert because lower MAE is better
    
    # --- Plot 2: MAE vs Model Size ---
    ax2 = axes[1]
    ax2.scatter(model_sizes, mae_scores, s=150, c='darkorange', edgecolors='black', zorder=3)
    
    for i, txt in enumerate(models):
        ax2.annotate(txt, (model_sizes[i], mae_scores[i]), 
                     xytext=(10, 5), textcoords='offset points', fontsize=10)
        
    ax2.set_title('Memory Efficiency (Accuracy vs Size)', fontsize=14)
    ax2.set_xlabel('Model Size (Millions of Parameters)', fontsize=12)
    ax2.set_ylabel('Weighted Mean Absolute Error', fontsize=12)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("model_comparison_plots.png", dpi=300, bbox_inches='tight')
    print("Successfully saved plots to 'model_comparison_plots.png'")

if __name__ == "__main__":
    plot_pareto_frontiers()