import torch
import numpy as np
import pandas as pd
from train_aux import DEVICE, SAVE_PATH
from backend.config import CLASS_ORDER, FEATURES, WINDOW_SIZE
from model.model_tcn import TCNModel


# Create a new model instance and load the saved state_dict
# Only load model once on import
MODEL = TCNModel().to(DEVICE)
MODEL.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
MODEL.eval() # Set to evaluation mode for predictions


def build_input(df: pd.DataFrame) -> torch.Tensor:
  """
  Builds input tensor for the model
  """
  x = df[FEATURES].tail(WINDOW_SIZE).to_numpy(dtype=np.float32)
  x = torch.from_numpy(x).unsqueeze(0)
  return x.to(DEVICE)

def pass_input(X: torch.Tensor) -> torch.Tensor:
  """
  Passes input tensor into the saved model
  """
  with torch.inference_mode():
    X = X.transpose(1, 2) # Transpose tensor to correct shape
    logits = MODEL(X)
    probs = torch.softmax(logits, dim=1) # Get probabilities for each class
    probs = probs.squeeze(0).tolist() # Remove the dimension for batch size and converts the tensor to a 2D List
  return {"probabilites": dict(zip(CLASS_ORDER, probs))} # Return probabilites for each class as a nested dictionary