from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

    
#* LSTM
class LSTM(nn.Module):
    """
    A customizable Long Short-Term Memory (LSTM) network for regression.

    Args:
        input_size (int):
            Number of input features.
        drop (float):
            Dropout probability for regularization.
        num_lstm_layers (int):
            Number of LSTM layers.
        delay (int):
            Number of time steps to look back in the input sequence.
        hidden_neurons (int, optional):
            Number of neurons in the LSTM layer.
    """
    def __init__(self, input_size: int, drop: float, num_lstm_layers: int, delay: int, hidden_neurons: int):
        super(LSTM, self).__init__()
        self.lstm_1 = nn.LSTM(
            input_size = input_size, hidden_size = hidden_neurons,
            num_layers = num_lstm_layers, batch_first = True, bidirectional = True
        )
        self.lstm_2 = nn.LSTM(
            input_size = hidden_neurons * 2, hidden_size = hidden_neurons, 
            num_layers = num_lstm_layers, batch_first = True, bidirectional = True
        )

        self.layer_norm = nn.LayerNorm(hidden_neurons * 2)

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_neurons * 2, 128), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, sequence_length, input_features]
        lstm_out_1, _ = self.lstm_1(x)
        lstm_out_2, _ = self.lstm_2(lstm_out_1)
        out_last = lstm_out_2[:, -1, :]
        out_norm = self.layer_norm(out_last)
        out = self.fc1(out_norm)
        return out
