import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, LSTM, Dense, TimeDistributed, Add, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from custom_loss_tf import hybrid_6dof_loss

def res_block(x, filters):
    """A custom mini-residual block that doesn't shrink the grid."""
    shortcut = x
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='linear', padding='same')(x)
    x = Add()([shortcut, x]) # The skip connection!
    return tf.keras.activations.relu(x)

def build_mini_resnet_lstm(window_size=20):
    print("Building Mini-ResNet-LSTM Architecture...")
    
    # 1. Build the isolated Spatial Engine first
    spatial_input = Input(shape=(8, 16, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(spatial_input)
    x = res_block(x, 32)
    x = res_block(x, 32)
    spatial_output = Flatten()(x)
    spatial_engine = Model(spatial_input, spatial_output, name="Mini_ResNet_Engine")
    
    # 2. Build the Temporal wrapper
    inputs = Input(shape=(window_size, 8, 16, 1))
    
    # Apply our custom ResNet engine to every frame in the window
    x = TimeDistributed(spatial_engine)(inputs)
    
    # 3. The Temporal Tracker
    x = LSTM(64, return_sequences=False)(x)
    outputs = Dense(6, activation='linear')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=hybrid_6dof_loss)
    return model