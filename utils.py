import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(WaveformClassifier, train_loader, val_loader, device, num_epochs=5):
  
  # Get the input shape
  x_batch, y_batch = next(iter(train_loader))
  input_shape = x_batch.shape
  model = WaveformClassifier( input_shape = input_shape).to(device)
  criterion = nn.CrossEntropyLoss().to(device)
  optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

  # Track metrics
  train_losses = []
  train_accs = []
  val_losses = []
  val_accs = []

  # Train
  for epoch in range(num_epochs):

    # Training loop
    model.train()
    train_epoch_loss = 0
    train_epoch_acc = 0

    for x_batch, y_batch in train_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)

      optimizer.zero_grad()
      y_pred = model(x_batch)
      loss = criterion(y_pred, y_batch)
      loss.backward()
      optimizer.step()

      train_epoch_loss += loss.item()
      train_epoch_acc += (y_pred.argmax(dim=1) == y_batch).sum().item() / len(y_batch)

    train_losses.append(train_epoch_loss / len(train_loader))
    train_accs.append(train_epoch_acc / len(train_loader))

    # Validation loop
    model.eval()
    val_epoch_loss = 0
    val_epoch_acc = 0

    with torch.inference_mode():
      for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        val_epoch_loss += loss.item()
        val_epoch_acc += (y_pred.argmax(dim=1) == y_batch).sum().item() / len(y_batch)

    val_losses.append(val_epoch_loss / len(val_loader))
    val_accs.append(val_epoch_acc / len(val_loader))

    # Print metrics
    print(f'Epoch {epoch+1}/{num_epochs}.. ',
          f'Train loss: {train_epoch_loss / len(train_loader):.4f}.. ',
          f'Val loss: {val_epoch_loss / len(val_loader):.4f}.. ',
          f'Val accuracy: {val_epoch_acc / len(val_loader):.4f}')
  return train_losses, train_accs, val_losses, val_accs




def train_with_validation(vae, train_loader, val_loader, optimizer, loss_fn, num_epochs, device):

  train_losses = []
  val_losses = []
  
  for epoch in range(num_epochs):

    vae.train()  
    total_train_loss = 0

    # Training loop
    for x_waveform, x_batch, _ in train_loader:
      x_batch = x_batch.to(device)
      
      optimizer.zero_grad()
      
      # Forward pass
      recon_batch, z_mean, z_logvar = vae(x_batch)
      
      # Compute training loss
      loss = loss_fn(recon_batch, x_batch, z_mean, z_logvar)
      
      # Backpropagation
      loss.backward()
      optimizer.step()
      
      total_train_loss += loss.item()
      
    # Calculate average training loss    
    average_train_loss = total_train_loss / len(train_loader)
    train_losses.append(average_train_loss)

    # Validation loop
    vae.eval()
    total_val_loss = 0
    
    with torch.inference_mode():
      for x_waveform, x_batch, _ in val_loader:
        x_batch = x_batch.to(device)
        
        # Forward pass
        recon_batch, z_mean, z_logvar = vae(x_batch)
        
        # Compute validation loss
        loss = loss_fn(recon_batch, x_batch, z_mean, z_logvar)
        total_val_loss += loss.item()
        
      # Calculate average validation loss
      average_val_loss = total_val_loss / len(val_loader)
      val_losses.append(average_val_loss)
            
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {average_train_loss:.4f} - Validation Loss: {average_val_loss:.4f}")
    # Save or use the trained VAE model for later inference
    torch.save(vae.state_dict(), './Exports/vae2deep_8_beta20.pth')

  return train_losses, val_losses

