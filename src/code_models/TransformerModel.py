import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


#* TRANSFORMER
class PositionalEncoding(nn.Module):
    """
    Implementation of Sinusoidal Positional Coding.
    Allows the Transformer to take sequence order into account.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encoding once in log space.
        pe = torch.zeros(max_len, d_model)          # Tensor to store the encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)         # (max_len, 1)

        # Division term for sinusoidal frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1) 
        pe = pe.squeeze(1).unsqueeze(0)
        self.register_buffer('pe', pe)      # Register 'pe' as a buffer, not as a trainable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.
        """
        # x shape: (batch_size, sequence_length, d_model)
        # self.pe shape: (1, max_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TRANSFORMER(nn.Module):
    """
    Transformer model for time series regression.
    """
    def __init__(self, input_features: int, d_model: int, nhead: int, num_encoder_layers: int, dim_feedforward: int, dropout: float, delay: int):
        super(TRANSFORMER, self).__init__()

        # Linear embedding layer to project input features to d_model
        self.input_embedding = nn.Linear(input_features, d_model)

        # Positional coding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len = delay + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = True  
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_encoder_layers)
        self.d_model = d_model 

        self.output_fc = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
            )

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.input_embedding.bias.data.zero_()
        self.output_fc.weight.data.uniform_(-initrange, initrange)
        self.output_fc.bias.data.zero_()

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the TransformerModel.
        Args:
            src (torch.Tensor): 
                Input (source) tensor of shape (batch_size, sequence_length, input_features).
            src_mask (Optional[torch.Tensor]): 
                Additive mask for self-attention in the encoder. Shape: (sequence_length, sequence_length).
            src_key_padding_mask (Optional[torch.Tensor]): 
                Boolean mask to ignore padding elements in `src`. Shape: (batch_size, sequence_length).
        """
        # src shape: (batch_size, sequence_length, input_features)
        src_embedded = self.input_embedding(src) * math.sqrt(self.d_model)      # src_embedded shape: (batch_size, sequence_length, d_model)

        # Add positional encoding
        src_pos_encoded = self.pos_encoder(src_embedded)

        # Passing through the Transformer Encoder
        # If batch_first=True is used, src_mask must be (N*num_heads, S, S) or (S,S) and src_key_padding_mask (N,S). For time series regression, a causal mask (src_mask) is often not needed unless you are doing stepwise autoregressive prediction *within* the model.
        # src_key_padding_mask is useful if the sequences in a batch have variable lengths and are padded.

        encoder_output = self.transformer_encoder(
            src_pos_encoded, 
            mask=src_mask, 
            src_key_padding_mask=src_key_padding_mask
        )

        # Take the output of the last time step (similar to LSTM/GRU) for prediction. This is a common way to obtain a fixed representation of the sequence for regression. Alternatives: use the [CLS] token if added, or apply pooling over the sequence.
        output_last_step = encoder_output[:, -1, :]

        out = self.output_fc(output_last_step)
        return out