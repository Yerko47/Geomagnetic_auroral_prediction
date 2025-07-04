from typing import Tuple, Dict, Any, Union, List

import torch
import torch.nn as nn

from code_models.ANN import ANN
from code_models.CNN import CNN
from code_models.LSTM import LSTM
from code_models.GRU import GRU
from code_models.TCNN import TCNN
from code_models.TransformerModel import TRANSFORMER

#* SELECTION MODEL
def type_nn(config: Dict[str, Any], x_train_shape: Tuple[int, ...], delay: int, device: Union[str, torch.device]) -> nn.Module:
    """
    Selects, instantiates, and configures the specified neural network model.

    The model type and its specific hyperparameters are read from the configuration dictionary (`config['nn']`). The input size for the model is inferred from `x_train_shape` based on conventions for each model type.

    Args:
        config (Dict[str, Any]): 
            Configuration dictionary. Expected keys under `config['nn']`:
                'type_model': (str) Name of the model (ANN, CNN, LSTM, GRU, TCNN, TRANSFORMER).
                'drop': (float) Dropout rate.
                'kernel_cnn': (int) Kernel size for CNN.
                'kernel_size_tcnn': (int) Kernel size for TCNN.
                'num_channels_list_tcnn': (List[int]) List of channels for TCNN blocks.
                'num_layer_lstm': (int) Number of layers for LSTM.
                'num_layer_gru': (int) Number of layers for GRU.
                'hidden_neurons_lstm': (int) Hidden neurons for LSTM.
                'hidden_neurons_gru': (int) Hidden neurons for GRU.
                'd_model_transformer': (int) Dimension for Transformer model.
                'nhead_transformer': (int) Number of heads for Transformer.
                'num_encoder_layers_transformer': (int) Number of encoder layers for Transformer.
                'dim_feedforward_transformer': (int) Dimension of feedforward network for Transformer.
        x_train_shape (Tuple[int, ...]): 
        The shape of one batch of training data features (e.g., train_solar.shape).
                                         Conventions based on `time_delay` output:
                                         - ANN: (batch_size, num_features_flattened)
                                         - CNN/TCNN: (batch_size, num_input_channels/features, sequence_length)
                                         - LSTM/GRU/Transformer: (batch_size, sequence_length, num_features)
        delay (int): 
            The sequence length. Corresponds to `delay_steps` from `time_delay`. This is `x_train_shape[2]` for CNN/TCNN and `x_train_shape[1]` for LSTM/GRU/Transformer.
        device (Union[str, torch.device]): 
            The device ('cuda', 'cpu', 'mps') for the model.

    Returns:
        model (nn.Module): 
            The instantiated PyTorch model, moved to the specified device.
    """

    model_config = config.get('nn', {})
    type_model = model_config.get('type_model').upper()

    drop = model_config.get('drop')

    # Extract ANN parameters
    hidden_neurons_ann = model_config.get('hidden_layer_ann')

    # Extract LSTM parameters
    num_lstm_layers = model_config.get('num_layer_lstm')
    hidden_neurons_lstm = model_config.get('hidden_neurons_lstm')

    # Extract CNN & TCNN parameters
    kernel_size = model_config.get('kernel_size')
    num_channels_list_tcnn = model_config.get('num_channel_list_tcnn')

    # Extract GRU parameters
    num_gru_layers = model_config.get('num_layer_gru')
    hidden_neurons_gru = model_config.get('hidden_neurons_gru')
    
    # Extract Transformer parameters
    default_d_model = x_train_shape[2] if type_model == 'TRANSFORMER' and len(x_train_shape) == 3 else 64
    d_model_transformer = model_config.get('d_model_transformer', default_d_model)
    nhead_transformer = model_config.get('nhead_transformer')
    num_encoder_layers_transformer = model_config.get('num_encoder_layers_transformer')
    dim_feedforward_transformer = model_config.get('dim_feedforward_transformer')
    if d_model_transformer % nhead_transformer != 0:
        raise ValueError(f"d_model_transformer ({d_model_transformer}) must be divisible by nhead_transformer ({nhead_transformer})")
    
    
    match type_model:

        case 'ANN':
            if len(x_train_shape) == 2:
                input_size = x_train_shape[1]
                model = ANN(input_size, hidden_neurons_ann, drop)
                print(f"\nInstantiated ANN model with input_size = {input_size}, hidden_neurons = {hidden_neurons_ann}")
    
        case 'LSTM':
            if len(x_train_shape) == 3:
                input_size = x_train_shape[2]
                model = LSTM(input_size, hidden_neurons_lstm, num_lstm_layers, drop, use_attention = False, num_ensemble_heads = 5)
                print(f"\nInstantiated LSTM model with input_features = {input_size}, layers = {num_lstm_layers}, hidden_neurons = {hidden_neurons_lstm}, sequence_length = {delay}")

        case 'GRU':
            if len(x_train_shape) == 3:
                input_size = x_train_shape[2]
                model = GRU(input_size, drop, num_gru_layers, delay, hidden_neurons_gru)
                print(f"\nInstantiated GRU model with input_features = {input_size}, layers = {num_gru_layers}, hidden_neurons = {hidden_neurons_gru}, sequence_length = {delay}")

        case 'TRANSFORMER':
            if len(x_train_shape) == 3:
                input_size = x_train_shape[2]
                model = TRANSFORMER(input_size, d_model_transformer, nhead_transformer, num_encoder_layers_transformer, dim_feedforward_transformer, drop, delay)
                print(f"\nInstantiated Transformer model with input_features = {input_size}, d_model = {d_model_transformer}, nhead = {nhead_transformer}, num_encoder_layers = {num_encoder_layers_transformer}, dim_feedforward = {dim_feedforward_transformer}, sequence_length = {delay}")

        case 'CNN':
            if len(x_train_shape) == 3:
                input_size = x_train_shape[1]
                model = CNN(input_size, kernel_size, drop, delay)
                print(f"\nInstantiated CNN model with input_channels = {input_size}, kernel_size = {kernel_size}, sequence_length = {delay}\n")


        case 'TCNN':
            if len(x_train_shape) == 3:
                input_size = x_train_shape[1]
                model = TCNN(input_size, num_channels_list_tcnn, kernel_size, drop, delay)
                print(f"\nInstantiated TCNN model with input_channels = {input_size}, kernel_size = {kernel_size}, sequence_length = {delay}")

        case _:
            raise ValueError(f"Invalid type_model: '{type_model}'. Choose from 'ANN', 'CNN', 'LSTM', 'GRU', 'TCNN', 'TRANSFORMER'.")
    
    if model is None:
        raise ValueError(f'Model instantiation failed for type: {type_model}')
    
    return model.to(device)