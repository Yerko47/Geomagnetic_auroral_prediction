from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#*ANN
class ANN(nn.Module):
    """
    A customizable Multilayer Perceptron (MLP) for regression.
    
    Args:
    input_size (int): 
        Number of input features.
    drop_rate (float): 
        Dropout probability for regularization.
    
    """
    def __init__(self, input_size: int, drop: float):
        super(ANN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 768), nn.ReLU(), nn.BatchNorm1d(768), nn.Dropout(drop),
            nn.Linear(768, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(drop),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(drop),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(drop),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc_layers(x)
        return(out)