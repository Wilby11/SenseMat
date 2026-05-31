import torch
import torch.nn as nn

def hybrid_6dof_loss(y_pred, y_true, lambda_rot=1.0):
    """
    Calculates the loss for Position and Rotation separately to ensure
    the network learns both physical properties equally well.
    """
    # Split the 6 outputs: [X, Y, Z] and [Pitch, Yaw, Roll]
    pos_pred, rot_pred = y_pred[:, :3], y_pred[:, 3:]
    pos_true, rot_true = y_true[:, :3], y_true[:, 3:]
    
    pos_loss = nn.functional.mse_loss(pos_pred, pos_true)
    rot_loss = nn.functional.mse_loss(rot_pred, rot_true)
    
    # Apply the lambda multiplier to rotation
    total_loss = pos_loss + (lambda_rot * rot_loss)
    return total_loss

def train_pytorch_model(model, train_loader, val_loader, optimizer, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            batch_y = batch_y[:, -1, :] # Slice the target tensor to only use the final frame's 6DoF label

            optimizer.zero_grad()           # Clear old gradients
            predictions = model(batch_x)    # Forward pass
            loss = hybrid_6dof_loss(predictions, batch_y, lambda_rot=1.0) # Calculate custom 6DoF loss
            
            loss.backward()                 # Backpropagation
            optimizer.step()                # Update weights
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
            
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad(): # Turn off gradients for validation!
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_y = batch_y[:, -1, :]
                predictions = model(batch_x)
                loss = hybrid_6dof_loss(predictions, batch_y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")