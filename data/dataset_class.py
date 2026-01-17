import torch
import numpy as np
"""
Create a dataset class by subclassing torch.utils.data.Dataset
"""
class IndexDataset(torch.utils.data.Dataset):
  def __init__(self, X: np.ndarray, y: np.ndarray):
    #Tensors for fetaures
    self.X = torch.tensor(X, dtype=torch.float32)
    #Tensors for labels
    self.y = torch.tensor(y, dtype=torch.long)
  
  """
  Returns the number of sequences
  """
  def __len__(self) -> int:
    return len(self.X)
  
  """
  Returns sample with index i  
  """
  def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
    return self.X[i], self.y[i]