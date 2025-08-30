import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from data.data_processing import get_train_test
from data.dataset_class import IndexDataset
from model.model_tcn import TCNModel


RANDOM_SEED = 99 # RNG seed
N_EPOCHS = 1000 # Number of loops to train for
BATCH_SIZE = 64 # Batch size for DataLoaders
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Sets device to GPU if available, CPU otherwise
# Setting RNG seed
torch.manual_seed(RANDOM_SEED)

def get_weights(train_labels: np.ndarray):
  """
  Function to calculate class weights for training data.

  Weights are inversely proportional to class frequency.
  """
  class_counts = np.bincount(train_labels, minlength=3) # Array of number of samples per class
  n_samples = class_counts.sum() # Total number of samples
  weights = n_samples / (len(class_counts) * class_counts) # Class weights
  class_weights  = torch.from_numpy(weights.astype("float32")) # Convert class weights to a Tensor

  return class_weights.to(DEVICE)

def get_loaders(return_weights=True) -> tuple:
  """
  Function to retrieve training and testing data and load it into DataLoaders

  If return_weights parameter is True, also returns class weights
  """
  X_train, X_test, y_train, y_test = get_train_test() # Gets training and testing data
  # Creates dataset objects from training and testing data
  train_dataset = IndexDataset(X_train, y_train)
  test_dataset = IndexDataset(X_test, y_test)
  # Creates DataLoaders from dataset objects
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

  # If return_weights is true, calculates and returns class weights, otherwise only returns DataLoaders
  if return_weights:
    class_weights = get_weights(y_train)
    return train_loader, test_loader, class_weights
  else:
    return train_loader, test_loader

def test_loop(model: TCNModel, loader, epoch: int, loss_fn, acc_metric):
  """
  Function to calculate and print the loss and accuracy of model predictions on test data
  """
  model.eval() # Sets the model to evaluation mode
  total_loss = 0.0
  n_samples = 0
  acc_metric.reset()
  # Use torch.inference_mode to disable gradient tracking
  with torch.inference_mode():
    # Calculates loss and accuracy using training Dataset
    for x_ts, y_ts in loader:
      x_ts, y_ts = x_ts.to(DEVICE), y_ts.to(DEVICE)
      x_ts = x_ts.transpose(1, 2)
      test_logits = model(x_ts)
      test_loss = loss_fn(test_logits, y_ts)
      total_loss += test_loss.item() * x_ts.size(0)
      n_samples += x_ts.size(0)

      test_preds = torch.argmax(test_logits, dim=1)
      acc_metric.update(test_preds, y_ts)
      
  print(f"Epoch: {epoch} | Test Loss: {(total_loss/n_samples):.5f} Test Accuracy: {(acc_metric.compute().item() * 100):.2f}")

def train_loop(model: TCNModel, epochs: int, acc_metric,):
  """
  Function to perform the training loop
  """
  train_loader, test_loader, weights = get_loaders() # Get DataLoaders and class weights
  loss_fn = torch.nn.CrossEntropyLoss(weight=weights) # Cross Entropy Loss function
  optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4, weight_decay=1e-3) # AdamW optimizer
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=N_EPOCHS) # Cosine Annealing scheduler

  for epoch in range(epochs):
    model.train() # Sets model to training mode
    total_loss = 0.0 # Stores cumulative loss for each batch
    n_samples = 0 # Number of samples per batch
    acc_metric.reset() # Resets accuracy metric

    for x_tr, y_tr in train_loader:
      x_tr, y_tr = x_tr.to(DEVICE), y_tr.to(DEVICE) # Move data from DataLoaders to correct device
      # Data is of shape (batch_size, window_length, num features)
      # Transpose the data to be of shape (batch_size, num features, window_length)
      x_tr = x_tr.transpose(1, 2)
      y_logits = model(x_tr) # Passes training data into the model
      loss = loss_fn(y_logits, y_tr) # Calculates training loss
      optimizer.zero_grad() # Reset the gradient of optimizer params
      loss.backward() # Backpropogate
      optimizer.step() # Gradient descent

      if scheduler is not None:
        scheduler.step() # Step scheduler if it exists

      # Update loss and accuracy
      total_loss += loss.item() * x_tr.size(0) # Loss is weighted by batch size
      n_samples += x_tr.size(0)
      y_preds = torch.argmax(y_logits, dim=1) # Get prediction labels from training data
      acc_metric.update(y_preds, y_tr) # Get model accuracy on training data
    
    # Prints the accuracy and loss on training and test data every 50 epochs
    if epoch % 50 == 0:
      print(f"Epoch: {epoch} | Learn Rate {optimizer.param_groups[0]["lr"]} | " 
            f"Train Loss: {(total_loss/n_samples):.5f} Train Accuracy: {(acc_metric.compute().item() * 100):.2f}")
      test_loop(model, test_loader, epoch, loss_fn, acc_metric)