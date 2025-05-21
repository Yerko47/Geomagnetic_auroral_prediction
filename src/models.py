import math
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
            nn.Linear(input_size, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(drop),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(drop),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(drop),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(drop),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(drop),
            nn.Linear(32, 16), nn.ReLU(), nn.BatchNorm1d(16), nn.Dropout(drop),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc_layers(x)
        return(out)
    

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
            ResBlock(input_channels = input_size, output_channels = 64, kernel_size = kernel_size, stride = 1, padding = padding),
            nn.MaxPool1d(kernel_size = 2, stride = 2), nn.Dropout(drop),
            ResBlock(input_channels = 64, output_channels = 128, kernel_size = kernel_size, stride = 1, padding = padding),
            nn.MaxPool1d(kernel_size = 2, stride = 2), nn.Dropout(drop),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(128 + input_size, 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(drop)
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
        self.lstm = nn.LSTM(
            input_size = input_size, hidden_size = hidden_neurons,
            num_layers = num_lstm_layers, batch_first = True, bidirectional = True
        )

        self.layer_norm = nn.LayerNorm(hidden_neurons * 2)

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_neurons * 2, 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, sequence_length, input_features]
        lstm_out, _ = self.lstm(x)
        out_last = lstm_out[:, -1, :]
        out_norm = self.layer_norm(out_last)
        out = self.fc1(out_norm)
        return out


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
            nn.Linear(hidden_neurons * 2, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)                   
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, sequence_length, input_features]
        gru_out, _ = self.gru(x)
        out_last_step = gru_out[:, -1, :]
        out_norm = self.layer_norm(out_last_step)
        out = self.fc(out_norm)
        return out
    

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
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride = stride, padding = padding, dilation = dilation))
        self.champ1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(drop)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride = stride, padding = padding, dilation = dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(drop)

        self.net = nn.Sequential(self.conv1, self.champ1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
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
        self.fc = nn.Linear(num_channels_list[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, input_channels, sequence_length]
        out_temporal = self.temporal_network(x)
        out_last = out_temporal[:, :, -1]
        out = self.fc(out_last)
        return out


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

class TransformerModel(nn.Module):
    """
    
    """
    def __init__(self, input_features: int, d_model: int, nhead: int, num_encoder_layers: int, dim_feedforward: int, dropout: float, delay: int):
        super(TransformerModel, self).__init__()

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

        self.output_fc = nn.Linear(d_model, 1)

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