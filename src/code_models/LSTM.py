import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Implementación de Multi-Head Attention.
    Esta parte se mantiene sin cambios por si se desea usar en el futuro.
    """
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.output(attended)

class ResidualBlock(nn.Module):
    """
    Bloque LSTM con conexiones residuales y normalización.
    Se mantiene para conservar la estructura escalable.
    """
    def __init__(self, input_size: int, hidden_size: int, drop: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.drop = nn.Dropout(drop)

        self.residual_projection = None
        if input_size != hidden_size * 2:
            self.residual_projection = nn.Linear(input_size, hidden_size * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        
        if self.residual_projection is not None:
            residual = self.residual_projection(x)
        else:
            residual = x
        
        # Aplicamos la conexión residual al último paso de tiempo
        lstm_out = residual + lstm_out

        return self.drop(self.layer_norm(lstm_out))
    
class LSTM(nn.Module):
    """
    Versión adaptada del modelo LSTM, combinando la robustez de la nueva
    arquitectura con la simplicidad del clasificador del modelo antiguo.
    """
    def __init__(self, input_size: int, hidden_neurons: int, num_lstm_layers: int, drop: float, use_attention: bool = False, num_ensemble_heads: int = 1):
        super(LSTM, self).__init__()

        self.use_attention = use_attention
        
        # Proyección de entrada, se mantiene para flexibilidad
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_neurons),
            nn.BatchNorm1d(hidden_neurons),
            nn.SiLU(),
            nn.Dropout(drop * 0.5)
        )

        self.lstm_blocks = nn.ModuleList()
        for i in range(num_lstm_layers):
            block_input_size = hidden_neurons if i == 0 else hidden_neurons * 2
            self.lstm_blocks.append(
                ResidualBlock(input_size=block_input_size, hidden_size=hidden_neurons, drop=drop)
            )
        
        if self.use_attention:
            self.attention = MultiHeadAttention(hidden_size=hidden_neurons * 2, num_heads=8)
            self.attention_norm = nn.LayerNorm(hidden_neurons * 2)
        
        # --- CAMBIO PRINCIPAL: Cabezas de predicción simplificadas ---
        # Esta estructura replica la de los modelos antiguos (ej. LSTM_1)
        self.ensemble_heads = nn.ModuleList()
        for _ in range(num_ensemble_heads):
            # La entrada es hidden_neurons * 2 debido a la LSTM bidireccional
            head = nn.Sequential(
                nn.Linear(hidden_neurons * 2, 128),
                nn.ReLU(),  # Usando ReLU como en el modelo antiguo
                nn.Dropout(drop),
                
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(drop),
                
                nn.Linear(64, 32),
                nn.ReLU(),
                
                nn.Linear(32, 1)
            )
            self.ensemble_heads.append(head)

        # Si solo usamos una cabeza, los pesos de ensamble no son necesarios
        if num_ensemble_heads > 1:
            self.ensemble_weights = nn.Parameter(torch.ones(num_ensemble_heads) / num_ensemble_heads)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        n = param.size(0)
                        param.data[n//4 : n//2].fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x_reshaped = x.view(-1, x.size(-1))
        x_proj = self.input_projection(x_reshaped)
        x = x_proj.view(batch_size, seq_len, -1)
        
        for lstm_block in self.lstm_blocks:
            x = lstm_block(x)
        
        if self.use_attention:
            attended = self.attention(x)
            x = self.attention_norm(x + attended)
        
        # --- CAMBIO PRINCIPAL: Usar solo la salida del último paso de tiempo ---
        final_repr = x[:, -1, :]  # Shape: (batch_size, hidden_neurons * 2)
        
        # Si hay múltiples cabezas, se calculan todas las predicciones
        if len(self.ensemble_heads) > 1:
            predictions = [head(final_repr) for head in self.ensemble_heads]
            ensemble_pred = torch.stack(predictions, dim=-1)
            weights = F.softmax(self.ensemble_weights, dim=0)
            final_pred = torch.sum(ensemble_pred * weights.view(1, 1, -1), dim=-1)
        else:
            # Si solo hay una cabeza, la predicción es directa
            final_pred = self.ensemble_heads[0](final_repr)
            
        return final_pred