import torch
from torchmetrics import Accuracy, Precision, Recall
from model.model_tcn import TCNModel
from train_aux import RANDOM_SEED, N_EPOCHS, PATIENCE, DEVICE, get_loaders # File with helper functions and constants

# Setting RNG seed
torch.manual_seed(RANDOM_SEED)

def test_loop(model: TCNModel, loader, loss_fn, accuracy, precision, recall):
  """
  Function to calculate and print the loss and accuracy of model predictions on test data
  """
  model.eval() # Sets the model to evaluation mode
  total_loss = 0.0
  n_samples = 0
  # Reset metrics
  accuracy.reset()
  precision.reset()
  recall.reset()
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
      # Get metrics on testing data
      accuracy.update(test_preds, y_ts)
      precision.update(test_preds, y_ts)
      recall.update(test_preds, y_ts)

      test_acc = accuracy.compute().item() * 100
      test_prec = precision.compute().item() * 100
      test_rec = recall.compute().item() * 100
      
  print(f"\nFINAL TEST\n"
        f"Loss: {(total_loss/n_samples):.5f} Accuracy: {test_acc:.2f} Precision: {test_prec:.2f} Recall {test_rec:.2f}")

def train_loop(model: TCNModel, epochs: int):
  """
  Function to perform the training loop
  """
  model.to(DEVICE) # Move model to correct device
  train_loader, val_loader, test_loader, weights = get_loaders() # Get DataLoaders and class weights
  loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1) # Cross Entropy Loss function
  optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4, weight_decay=1e-3) # AdamW optimizer
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=N_EPOCHS) # Cosine Annealing scheduler

  # Torchmetrics metrics
  accuracy = Accuracy(task="multiclass", num_classes=3).to(DEVICE)
  precision = Precision(task="multiclass", num_classes=3).to(DEVICE)
  recall = Recall(task="multiclass", num_classes=3).to(DEVICE)

  best_state = None # Model state with the best validation loss
  best_loss = float("inf") # Lowest validation loss
  wait_epochs = 0 # Number of epochs since validation loss improvement

  """Training"""
  for epoch in range(1, epochs + 1):
    model.train() # Sets model to training mode
    total_train_loss = 0.0 # Stores cumulative loss for each batch
    n_train_samples = 0 # Number of samples per batch
    # Reset metrics
    accuracy.reset()
    precision.reset()
    recall.reset()

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
      total_train_loss += loss.item() * x_tr.size(0) # Loss is weighted by batch size
      n_train_samples += x_tr.size(0)
      y_preds = torch.argmax(y_logits, dim=1) # Get prediction labels from training data
      accuracy.update(y_preds, y_tr) # Get model accuracy on training data
      precision.update(y_preds, y_tr) # Get precision accuracy on training data
      recall.update(y_preds, y_tr) # Get model recall on training data
    
    train_loss = total_train_loss/n_train_samples
    train_acc, train_prec, train_rec = accuracy.compute().item() * 100, precision.compute().item() * 100, recall.compute().item() * 100

    """Validation"""
    model.eval()
    # Reset metrics
    accuracy.reset()
    precision.reset()
    recall.reset()
    total_val_loss, n_val_samples = 0.0, 0

    with torch.inference_mode(): # Disable gradient tracking
      for x_vl, y_vl in val_loader:
        x_vl, y_vl = x_vl.to(DEVICE), y_vl.to(DEVICE)
        x_vl = x_vl.transpose(1, 2)
        val_logits = model(x_vl)
        val_loss = loss_fn(val_logits, y_vl)
        total_val_loss = val_loss.item() * x_vl.size(0)
        n_val_samples += x_vl.size(0)
        val_preds = torch.argmax(val_logits, dim=1)
        accuracy.update(val_preds, y_vl)
        precision.update(val_preds, y_vl)
        recall.update(val_preds, y_vl)
      
    val_loss = total_val_loss/n_val_samples
    val_acc, val_prec, val_rec = accuracy.compute().item() * 100, precision.compute().item() * 100, recall.compute().item() * 100

    scheduler.step() # Step the scheduler
    
    # Prints the accuracy and loss on training and test data every 50 epochs
    if epoch % 50 == 0 or epoch == 1:
      print(f"Epoch: {epoch} | Learn Rate {optimizer.param_groups[0]["lr"]}" 
            f"\nTraining | Loss: {train_loss:.5f} Accuracy: {train_acc:.2f} Precision: {train_prec:.2f} Recall: {train_rec:.2f}"
            f"\nValidation | Loss: {val_loss:5f} Accuracy: {val_acc:.2f} Precision: {val_prec:.2f} Recall: {val_rec:.2f}")
    
    """Detect overfitting using validation loss"""
    if val_loss < best_loss - 1e-6: # If no overfitting is detected, update the best validation loss and wait
      best_loss = val_loss
      wait_epochs = 0
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} # Stores a model state with no overfitting
    else: # If the validation loss increases after an epoch, increment the wait
      wait_epochs += 1
      if wait_epochs >= PATIENCE: # If the wait epochs reach the patience limit, end training early
        print(f"Stopping training early at Epoch {epoch} (Best Validation Loss: {best_loss:.5f})")
        break
  
  #Load the best model state for testing
  if best_state is not None:
    model.load_state_dict(best_state)
  
  """Testing"""
  test_loop(model, test_loader, loss_fn, accuracy, precision, recall) # Run the best model state on the test 