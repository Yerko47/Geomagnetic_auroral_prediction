from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

    
#* GRU
class GRU(nn.Module):
    """
    A customizable Gated Recurrent Unit (GRU) network for regression.

    Args:
        input_size (int):
            Number of input features.
        drop (float):
            Dropout probability for regularization.
        num_gru_layer (int):
            Number of GRU layers.
        delay (int):
            Number of time steps to look back in the input sequence.
        hidden_neurons (int, optional):
            Number of neurons in the GRU layer.
    """
    def __init__(self, input_size: int, drop: float, num_gru_layer: int, delay: int, hidden_neurons: int):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size = input_size, hidden_size = hidden_neurons,
            num_layers = num_gru_layer, batch_first = True, bidirectional = True
            ) 
        
        self.layer_norm = nn.LayerNorm(hidden_neurons * 2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_neurons * 2, 128), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, 1)                   
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, sequence_length, input_features]
        gru_out, _ = self.gru(x)
        out_last_step = gru_out[:, -1, :]
        out_norm = self.layer_norm(out_last_step)
        out = self.fc(out_norm)
        return out