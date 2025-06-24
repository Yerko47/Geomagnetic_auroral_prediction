import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


#* TCNN 
class Chomp1d(nn.Module):
    """
    Module for chomping elements from the end of a time sequence.
    Used to ensure causality in convolutions where the padding is symmetrical.
    """
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        if self.chomp_size == 0:
            return x
        return x[:, :, : - self.chomp_size].contiguous()
    

class TemporalBlock(nn.Module):
    """
    Optimized Temporal Block for large-scale data processing.
    Uses depthwise separable convolutions for efficiency.
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, drop: float, stride: int = 1, dilation: int = 1, padding: int = 0, use_separable: bool = True):
        super(TemporalBlock, self).__init__()

        self.use_separable = use_separable

        if use_separable and n_inputs == n_outputs:
            # Depthwise separable convolution for efficiency
            self.conv1 = nn.Sequential(
                nn.Conv1d(n_inputs, n_inputs, kernel_size, stride = stride, padding = padding, dilation = dilation, groups = n_inputs),
                nn.Conv1d (n_inputs, n_outputs, 1)
            )

            self.conv2 = nn.Sequential(
                nn.Conv1d(n_outputs, n_outputs, kernel_size, stride = stride, padding = padding, dilation = dilation, groups = n_outputs),
                nn.Conv1d(n_outputs, n_outputs, 1)
            )
        else:
            # Standard convolution
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride = stride, padding = padding, dilation = dilation)
            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride = stride, padding = padding, dilation = dilation)

        self.chomp1 = Chomp1d(padding)
        self.chomp2 = Chomp1d(padding)

        self.activation = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.bn2 = nn.BatchNorm1d(n_outputs)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.init_weights()

    def init_weights(self):
        """
        Initialization for deep networks
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)

        return self.activation(out + res)
    

class TCNN(nn.Module):
    """
    Temporal Convolutional Neural Network (TCNN) for time series regression.
    Consists of a stack of TemporalBlocks.
    Args:
        input_size (int): Number of input features.
        num_channels (List[int]): List with the number of output channels for each TemporalBlock.
                                  The length of this list determines the number of TemporalBlocks.
        kernel_size (int): Kernel size for the convolutions in the TemporalBlocks.
        dropout_rate (float): Dropout rate.
        sequence_length (int): Length of the input sequence. (Not used directly in the layer definitions if layers handle variable-length sequences, but may be useful for the final layer or to infer other dimensions).
        output_size (int, optional): Size of the final output. Defaults to 1 (for regression).
    """

    def __init__(self, input_size: int, num_channels_list_tcnn: List[int], kernel_size: int, drop: float, delay: int, use_separable: bool = True):
        super(TCNN, self).__init__()

        self.delay = delay

        layers = []
        num_levels = len(num_channels_list_tcnn)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels_list_tcnn[i - 1]
            out_channels = num_channels_list_tcnn[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(n_inputs = in_channels, n_outputs = out_channels, kernel_size = kernel_size,
                              stride = 1, dilation = dilation_size, padding = padding, drop = drop, 
                              use_separable = use_separable)
            )

        self.temporal_network = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(num_channels_list_tcnn[-1], 128), nn.GELU(), nn.Dropout(drop),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(drop * 0.5),
            nn.Linear(64, 1)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def get_receptive_field(self, kernel_size: int, num_levels: int) -> int:
        """
        Calculate the receptive field of the network
        """
        receptive_field = 1
        for i in range(num_levels):
            dilation = 2 * i
            receptive_field += (kernel_size - 1) * dilation
        return receptive_field

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            x: Input tensor of shape [batch_size, input_size, sequence_length]
        
        Returns:
            Prediction tensor of shape [batch_size, 1]
        """

        temporal_out = self.temporal_network(x)

        if temporal_out.size(-1) > 1:
            final_features = temporal_out[:, :, -1]
        else:
            final_features = self.adaptive_pool(temporal_out).squeeze(-1)
        
        out = self.fc(final_features)
        return out
