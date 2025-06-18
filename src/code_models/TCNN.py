import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


#* TCNN 
# (Inspired by https://github.com/LXP-Never/TCNN y Bai et al., 2018)
class Chomp1d(nn.Module):
    """
    Module for chomping elements from the end of a time sequence.
    Used to ensure causality in convolutions where the padding is symmetrical.
    """
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 
                Input tensor of shape (batch, channels, sequence_length).
        
        Returns:
            torch.Tensor: 
                Tensor with `chomp_size` elements removed from the end of the sequence dimension.
        """
        return x[:, :, : - self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    Individual block of a Temporal Convolutional Network (TCNN). It consists of two dilated and causal 1D convolutional layers, with weight normalization (WeightNorm), ReLU activation, and Dropout. It includes a residual connection.
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, drop: float = 0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride = stride, padding = padding, dilation = dilation)
            )
        self.champ1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(drop)

        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride = stride, padding = padding, dilation = dilation)
            )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(drop)

        self.conv3 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride = stride, padding = padding, dilation = dilation)
            )
        self.chomp3 = Chomp1d(padding)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(drop)


        self.net = nn.Sequential(self.conv1, self.champ1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2,
                                 self.conv3, self.chomp3, self.relu3, self.dropout3)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    
class TCNN(nn.Module):
    """
    Temporal Convolutional Neural Network (TCNN) for time series regression.
    Consists of a stack of TemporalBlocks.
    Args:
        input_channels (int): 
            Number of input features.
        num_channels (List[int]): List with the number of output channels for each TemporalBlock.
                                  The length of this list determines the number of TemporalBlocks.
        kernel_size (int): Kernel size for the convolutions in the TemporalBlocks.
        dropout_rate (float): Dropout rate.
        sequence_length (int): Length of the input sequence. (Not used directly in the layer definitions if layers handle variable-length sequences, but may be useful for the final layer or to infer other dimensions).
        output_size (int, optional): Size of the final output. Defaults to 1 (for regression).
    """
    def __init__(self, input_channels: int, num_channels_list: List[int], kernel_size: int, dropout_rate: float, sequence_length: int, output_size: int = 1):
        super(TCNN, self).__init__()
        layers = []
        num_levels = len(num_channels_list)         # Number of TemporalBlocks

        for i in range(num_levels):
            dilation_size = 2 ** i          # Exponential Dilation
            in_channels_block = input_channels if i == 0 else num_channels_list[i - 1]
            out_channels_block = num_channels_list[i]

            # Padding for causal convolution: (kernel_size - 1) * dilation_size
            # This ensures that the convolution only sees past points and the current one.
            # Chomp1d will take care of removing the padding on the right side.
            current_padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels_block, out_channels_block, kernel_size, stride = 1, dilation = dilation_size, padding = current_padding, drop = dropout_rate))

        self.temporal_network = nn.Sequential(*layers)

        # Final linear layer to map the output of the last TemporalBlock to the prediction.
        # The output of the last time step of the last block is taken.
        # The number of input features to this layer is the number of channels in the last block.
        self.fc = nn.Sequential(
            nn.Linear(num_channels_list[-1], 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, input_channels, sequence_length]
        out_temporal = self.temporal_network(x)
        out_last = out_temporal[:, :, -1]
        out = self.fc(out_last)
        return out