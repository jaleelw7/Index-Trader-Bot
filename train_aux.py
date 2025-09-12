import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from data.data_processing import get_train_test_val
from data.dataset_class import IndexDataset
from model.model_tcn import TCNModel



RANDOM_SEED = 99 # RNG seed
N_EPOCHS = 1000 # Number of loops to train 
PATIENCE = 5 # Number of epochs to wait for validation loss improvement
BATCH_SIZE = 64 # Batch size for DataLoaders
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Sets device to GPU if available, CPU otherwise
SAVE_DIR = Path("artifacts/models")
MODEL_FILE = "index_tcn_v1.pth"

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
  X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_val() # Gets training, validation and testing data
  # Creates dataset objects from training, validation and testing data
  train_dataset = IndexDataset(X_train, y_train)
  val_dataset = IndexDataset(X_val, y_val)
  test_dataset = IndexDataset(X_test, y_test)
  # Creates DataLoaders from dataset objects
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

  # If return_weights is true, calculates and returns class weights, otherwise only returns DataLoaders
  if return_weights:
    class_weights = get_weights(y_train)
    return train_loader, val_loader, test_loader, class_weights
  else:
    return train_loader, val_loader, test_loader

def save_model(model: TCNModel):
  """
  Function to save the model to artifacts/models directory
  """
  SAVE_DIR.mkdir(parents=True, exist_ok=True)
  SAVE_PATH = SAVE_DIR / MODEL_FILE
  try:
    torch.save(obj=model.state_dict(), f=SAVE_PATH)
    print(f"Model saved to {SAVE_PATH} successfully.")
  except FileNotFoundError:
    print(f"Directory not found at {SAVE_DIR}. Please check path and permissions.")
  except PermissionError:
    print(f"Permission denied when trying to save model to {SAVE_PATH}")
