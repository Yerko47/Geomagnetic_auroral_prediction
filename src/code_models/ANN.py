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
    def __init__(self, input_size: int, hidden_layer_ann: int, drop: float):
        super(ANN, self).__init__()

        layers = []
        hidden_layer_ann = int(hidden_layer_ann)
        while hidden_layer_ann > 10:
            layers.append(nn.Linear(input_size, hidden_layer_ann))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_layer_ann))
            layers.append(nn.Dropout(drop))
            input_size = hidden_layer_ann
            hidden_layer_ann = hidden_layer_ann // 2

            if hidden_layer_ann <= 256:
                drop = drop * 0.9

        self.temporal_network = nn.Sequential(*layers)
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.temporal_network(x)
        out = self.fc(out)
        return(out)
