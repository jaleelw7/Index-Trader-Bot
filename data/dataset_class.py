import torch
import numpy as np

class IndexDataset(torch.utils.data.Dataset):
  """
  A dataset class that subclasses torch.utils.data.Dataset. Allows use of PyTorch DataLoaders

  Attributes:
      X (torch.Tensor): A tensor of features
      y (torch.Tensor): A tensor of labels
  """
  def __init__(self, X: np.ndarray, y: np.ndarray):
    """
    Initializes a new IndexDataset instance. Creates tensors from ndarrays of features and labels

    Args:
      X (np.ndarray): Array of features
      y (np.ndarray): Array of labels
    """
    #Tensors for fetaures
    self.X = torch.tensor(X, dtype=torch.float32)
    #Tensors for labels
    self.y = torch.tensor(y, dtype=torch.long)
  

  def __len__(self) -> int:
    """
    Returns the number of time sequences

    Returns:
        int: The number of time sequences in the data
    """
    return len(self.X)
  

  def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns sample with index i

    Args:
        i (int): The index of a data sample in the dataset
    
    Returns:
        tuple[torch.Tensor,torch.Tensor]: The feature and label tensors for the sample with index i
    """
    return self.X[i], self.y[i]