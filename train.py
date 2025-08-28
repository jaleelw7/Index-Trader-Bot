import torch
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

# Metric from torchmetrics; Measures model accuracy
accuracy = Accuracy(task="multiclass", num_classes=3).to(DEVICE)

def test_loop(model: TCNModel, loader, epoch: int, loss_fn):
  """
  Function to calculate and print the loss and accuracy of model predictions on test data
  """
  model.eval() # Sets the model to evaluation mode
  total_loss = 0.0
  n_samples = 0
  accuracy.reset()
  # Use torch.inference_mode to disable gradient tracking
  with torch.inference_mode():
    for x_ts, y_ts in loader:
      x_ts, y_ts = x_ts.to(DEVICE), y_ts.to(DEVICE)
      x_ts = x_ts.transpose(1, 2)
      test_logits = model(x_ts)
      test_loss = loss_fn(test_logits, y_ts)
      total_loss += test_loss.item() * x_ts.size(0)
      n_samples += x_ts.size(0)

      test_preds = torch.argmax(test_logits, dim=1)
      accuracy.update(test_preds, y_ts)
      
  print(f"Epoch: {epoch} | Test Loss: {(total_loss/n_samples):.5f} Test Accuracy: {accuracy.compute().item():.4f}")

def train_loop(model: TCNModel, epochs: int, optimizer, loss_fn):
  """
  Function to perform the training loop
  """
  X_train, X_test, y_train, y_test = get_train_test() # Gets training and testing data
  # Creates dataset objects from training and testing data
  train_dataset = IndexDataset(X_train, y_train)
  test_dataset = IndexDataset(X_test, y_test)
  # Creates DataLoaders from dataset objects
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
  """
  Training Loop
  """
  for epoch in range(epochs):
    model.train() # Sets model to training mode
    total_loss = 0.0 # Stores cumulative loss for each batch
    n_samples = 0 # Number of samples per batch
    accuracy.reset() # Resets accuracy metric

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

      # Update loss and accuracy
      total_loss += loss.item() * x_tr.size(0) # Loss is weighted by batch size
      n_samples += x_tr.size(0)
      y_preds = torch.argmax(y_logits, dim=1) # Get prediction labels from training data
      accuracy.update(y_preds, y_tr) # Get model accuracy on training data
    
    # Prints the accuracy and loss on training and test data every 100 epochs
    if epoch % 10 == 0:
      print(f"Epoch: {epoch} | Train Loss: {(total_loss/n_samples):.5f} Train Accuracy: {accuracy.compute().item():.4f}")
      test_loop(model, test_loader, epoch, loss_fn)

model_0 = TCNModel(in_size=8,
                   n_filters=[16, 16, 32],
                   kernel_sizes=[3, 3, 3],
                   dilations=[1, 2, 4],
                   n_classes=3).to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model_0.parameters(), lr=1e-3, weight_decay=1e-4)
train_loop(model_0, N_EPOCHS, optimizer=optimizer, loss_fn=loss_fn)