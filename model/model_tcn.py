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
    
  def forward(self, x):
    """
    Method that defines the forward computation on input
    """
    out = self.drop1(self.act1(self.conv1(self.pad1(x)))) # First layer forward pass
    out = self.drop2(self.act2(self.conv2(self.pad2(out)))) # Second layer forward pass
    
    # If input (x) and output channel dimensions are not equal, sets the input to the correct dimensions
    if self.identity_conv is not None:
      x = self.identity_conv(x)
    
    return nn.LeakyReLU(x + out)