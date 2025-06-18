from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

    

#* CNN + ResBlock
class ResBlock(nn.Module):
    """
    Residual Block for CNN
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, stride: int, padding: int):
        super(ResBlock, self).__init__()
    
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size, stride = stride, padding = padding, bias = False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace = False)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size, stride = 1, padding = padding, bias = False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace = False)
        )

        self.downsample = nn.Sequential() 
        if stride != 1 or input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size = 1, stride = stride, padding = 0, bias = False),
                nn.BatchNorm1d(output_channels)
            )

        self.final_relu = nn.ReLU(inplace = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.layer1(x)
        out = self.layer2(out)

        if out.size(2) != identity.size(2): 
            diff = out.size(2) - identity.size(2)
            if diff > 0: 
                identity = F.pad(identity, (diff // 2, diff - (diff // 2)))
            elif diff < 0: 
                out = F.pad(out, (-diff // 2, -diff + (-diff // 2))) 

        out = out + identity
        out = self.final_relu(out)
        
        return out

class CNN(nn.Module):
    """
    Customizable Convolutional Neural Network (CNN) for regression. 

    Args:
        input_size (int):
            Number of imput features.
        kernel_size (int):
            Size of the convolutional kernel.
        drop (float):
            Dropout probability for regularization.
        delay (int):
            Number of time steps to look back in the input sequence.
    """
    def __init__(self, input_size: int, kernel_size: int, drop: float, delay: int):
        super(CNN, self).__init__()
        padding = kernel_size // 2
        self.conv_blocks = nn.Sequential(
            ResBlock(input_channels = input_size, output_channels = 256, kernel_size = kernel_size, stride = 1, padding = padding),
            nn.MaxPool1d(kernel_size = 2, stride = 2), nn.Dropout(drop),
            ResBlock(input_channels = 256, output_channels = 512, kernel_size = kernel_size, stride = 1, padding = padding),
            nn.MaxPool1d(kernel_size = 2, stride = 2), nn.Dropout(drop),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(512 + input_size, 128), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(drop)
        )

        self.fc_out = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, input_features, sequence_length]
        x_last = x[:, :, -1]
        x_conv = self.conv_blocks(x)
        x_pooled = x_conv.mean(dim = -1)
        x_combined = torch.cat([x_pooled, x_last], dim = 1)
        out = self.fc1(x_combined)
        out = self.fc_out(out)
        return out
