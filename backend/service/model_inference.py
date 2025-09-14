import torch
from train_aux import DEVICE, SAVE_PATH
from backend.config import CLASS_ORDER
from model.model_tcn import TCNModel


# Create a new model instance and load the saved state_dict
# Only load model once on import
MODEL = TCNModel().to(DEVICE)
MODEL.load_state_dict(torch.load(SAVE_PATH, weights_only=True, map_location=DEVICE))
MODEL.eval() # Set to evaluation mode for predictions

def pass_input(X: torch.Tensor) -> dict:
  """
  Passes input tensor into the saved model
  """
  X = X.to(DEVICE) # Move tensor to correct device

  with torch.inference_mode():
    X = X.transpose(1, 2) # Transpose tensor to correct shape
    logits = MODEL(X)
    probs = torch.softmax(logits, dim=1) # Get probabilities for each class
    probs = probs.squeeze(0).tolist() # Remove the dimension for batch size and converts the tensor to a 2D List
  return {"probabilites": dict(zip(CLASS_ORDER, probs))} # Return probabilites for each class as a nested dictionary