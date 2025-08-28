import torch
from torch import nn

class ResidualBlock(nn.Module):
  """
  Class for TCN residual blocks subclassing nn.Module, with two dilated convolutional layers.

  Each convolutional layer uses the same dilation. Padding is only performed on the left to preserve causality.
  """
  def __init__(self,
               in_size: int,
               out_size: int,
               dilation: int,
               kernel_size: int,
               dropout: float = 0.2,
               slope: float = 0.01):
    super().__init__()
    self.input_size = in_size
    self.output_size = out_size

    pad_size = (kernel_size - 1) * dilation # Size of padding for conv layers

    self.pad1 = nn.ConstantPad1d((pad_size, 0), 0.0) # Padding for first conv layer
    # First conv layer
    self.conv1 = nn.utils.parametrizations.weight_norm(
      nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, dilation=dilation)
    )
    self.act1 = nn.LeakyReLU(negative_slope=slope) # Non-linear activation function for first layer
    self.drop1 = nn.Dropout1d(p=dropout) # Dropout for the first layer

    self.pad2 = nn.ConstantPad1d((pad_size, 0), 0.0) # Padding for second conv layer
    # Second conv layer
    self.conv2 = nn.utils.parametrizations.weight_norm(
      nn.Conv1d(in_channels=out_size, out_channels=out_size, kernel_size=kernel_size, dilation=dilation)
    )
    self.act2 = nn.LeakyReLU(negative_slope=slope) # Non-linear activation function for second layer
    self.drop2 = nn.Dropout1d(p=dropout) # Dropout for the second layer

    # Projects the input into the same dimensions as the output if their channel dimensions are different
    self.identity_conv = None
    if in_size != out_size:
      self.identity_conv = nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=1)
    
    self.final_act = nn.LeakyReLU(negative_slope=slope)
    
  def forward(self, x):
    """
    Method that defines the forward computation on input
    """
    # First layer forward pass
    out = self.pad1(x)
    out = self.conv1(out)
    out = self.act1(out)
    out = self.drop1(out)

    # Second layer forward pass
    out = self.pad2(out)
    out = self.conv2(out)
    out = self.act2(out)
    out = self.drop2(out)
    
    # If input (x) and output channel dimensions are not equal, sets the input to the correct dimensions
    if self.identity_conv is not None:
      x = self.identity_conv(x)
    
    return self.final_act(x + out)

class TCNModel(nn.Module):
  """
  Class for TCN model subclassing nn.Module, made from a stack of ResidualBlock objects.

  Predicts one label per time window
  """
  def __init__(self, 
               in_size: int, 
               n_filters: list[int], 
               kernel_sizes: list[int], 
               dilations: list[int],
               n_classes: int):
    super().__init__()
    # Checks if n_filters, kernel_sizes, and dilations have the same length. Raises a ValueError if False
    if not(len(n_filters) == len(kernel_sizes) == len(dilations)):
      raise ValueError("n_filters, kernel_sizes, and dilations must have the same length")
    
    res_blocks = [] # List of residual blocks
    # Adds each residual block to the list
    for i in range(len(n_filters)):
      in_c = in_size if i == 0 else n_filters[i-1] # Number of input channels for block i
      out_c = n_filters[i] # Number of output channels for block i
      res_blocks.append(ResidualBlock(in_size=in_c, out_size=out_c, kernel_size=kernel_sizes[i], dilation=dilations[i])) # Adds block i to the list
    
    self.residual_blocks = nn.Sequential(*res_blocks) # Integrates residual blocks into the model
    self.head = nn.Conv1d(in_channels=n_filters[-1], out_channels=n_classes, kernel_size=1) # Maps ResidualBlock output to classes
  
  def forward(self, x):
    """
    Method that defines the forward computation on input
    """
    out = self.residual_blocks(x)
    logit_seq = self.head(out) # Model output per timestep in input sequence
    return logit_seq[:, :, -1] # Returns model output for the last timestep only
