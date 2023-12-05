import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from tqdm import trange, tqdm
from pyitlib import discrete_random_variable as drv

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
    torch.save(vae.state_dict(), './Exports/betavae2deep_8.pth')

  return train_losses, val_losses

def train_with_validation_general(vae, train_loader, val_loader, optimizer, loss_fn, num_epochs, device,idx = 0, filename = "/Exports/betavae2deep_8.pth"):

  train_losses = []
  val_losses = []
  
  for epoch in range(num_epochs):

    vae.train()  
    total_train_loss = 0

    # Training loop
    for x_waveform, x_batch, _ in train_loader:
      x_batch = x_batch.to(device)
      # # Forward pass
      # recon_batch, z_mean, z_logvar = vae(x_batch)
      
      # # Compute training loss
      # loss = loss_fn(recon_batch, x_batch, z_mean, z_logvar)
      if idx == 3:
           loss = loss_fn(x_batch, vae, optimizer)
      else:
          # Forward pass
          optimizer.zero_grad()
          recon_batch, z_mean, z_logvar = vae(x_batch)
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
        
        # Compute validation loss
        if idx == 3:
           loss = loss_fn(x_batch, vae, optimizer)
        else:
          # Forward pass
          recon_batch, z_mean, z_logvar = vae(x_batch)
          loss = loss_fn(recon_batch, x_batch, z_mean, z_logvar)
        total_val_loss += loss.item()
        
      # Calculate average validation loss
      average_val_loss = total_val_loss / len(val_loader)
      val_losses.append(average_val_loss)
            
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {average_train_loss:.4f} - Validation Loss: {average_val_loss:.4f}")
    # Save or use the trained VAE model for later inference
    torch.save(vae.state_dict(), filename)

  return train_losses, val_losses

def get_bin_index(x, nb_bins):
    ''' Discretize input variable, adapted from https://github.com/ubisoft/ubisoft-laforge-disentanglement-metrics/blob/main/src/utils.py
    
    :param x:           input variable
    :param nb_bins:     number of bins to use for discretization

    '''
    # get bins limits
    bins = np.linspace(0, 1, nb_bins + 1)

    # discretize input variable
    return np.digitize(x, bins[:-1], right=False).astype(int)

def get_mutual_information(x, y, normalize=True):
    ''' Compute mutual information between two random variables, adapted from https://github.com/ubisoft/ubisoft-laforge-disentanglement-metrics/blob/main/src/utils.py
    
    :param x:      random variable
    :param y:      random variable
    '''
    if normalize:
        return drv.information_mutual_normalised(x, y, norm_factor='Y', cartesian_product=True)
    else:
        return drv.information_mutual(x, y, cartesian_product=True)
  


def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).

    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).

    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).

    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian.

    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.

    mu: torch.Tensor or np.ndarray or float
        Mean.

    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix

    Parameters
    ----------
    batch_size: int
        number of training images in the batch

    dataset_size: int
    number of training images in the dataset
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()



def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {nn.LeakyReLU: "leaky_relu", nn.ReLU: "relu", nn.Tanh: "tanh",
              nn.Sigmoid: "sigmoid", nn.Softmax: "sigmoid"}
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain


def linear_init(layer, activation="relu"):
    """Initialize a linear layer.
    Args:
        layer (nn.Linear): parameters to initialize.
        activation (`torch.nn.modules.activation` or str, optional) activation that
            will be used on the `layer`.
    """
    x = layer.weight

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity='leaky_relu')
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity='relu')
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def weights_init(module):
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        # TO-DO: check litterature
        linear_init(module)
    elif isinstance(module, nn.Linear):
        linear_init(module)
