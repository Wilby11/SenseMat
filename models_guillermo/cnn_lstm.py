import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, LSTM, Dense, TimeDistributed, Add, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from custom_loss_tf import hybrid_6dof_loss

def build_cnn_lstm(window_size=20):
    print("Building CNN-LSTM Architecture...")
    
    # 1. The Input Window
    inputs = Input(shape=(window_size, 8, 16, 1))
    
    # 2. The Spatial Extractor (CNN)
    # Stride=1 and padding='same' ensures we don't crush the 8x16 grid!
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    
    # Flatten the 8x16 feature maps into a 1D array for the LSTM
    x = TimeDistributed(Flatten())(x)
    
    # 3. The Temporal Tracker (LSTM)
    # The LSTM learns the momentum over the 20 frames
    x = LSTM(64, return_sequences=False)(x)
    
    # 4. The Output Layer (6 Linear Nodes for 6DoF)
    outputs = Dense(6, activation='linear')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=hybrid_6dof_loss)
    return model