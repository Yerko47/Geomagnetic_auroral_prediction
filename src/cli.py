import argparse

#* OVERRIDE CONFIGURATION VALUES
def parse_args():
    """
    Processed command-line arguments to override configuration values.

    Returns:
        overrides: dict
            Dictionary with keys and values that override config.yaml
    """
    parser = argparse.ArgumentParser(
        description = "Auroral Index Prediction: Overwrites config without editing files")
    
    # Data
    parser.add_argument('--auroral_index', type = str, help = 'Auroral Index Name [AE, AL, AU]')
    parser.add_argument('--scaler_type', type = str, help = "Scaler Type Data [robust, standard, minmax]")
    parser.add_argument('--set_split', type = str, help = 'Split Data [organized, random]')
    parser.add_argument('--test_size', type = float, help = "Proportion of test set [float]")

    # Cross Validation
    parser.add_argument('--n_split', type = int, help = "Number of Folds")
    parser.add_argument('--gap', type = int, help = "Optional: Gap between train and val on each fold")
    parser.add_argument('--max_train_size', type = int, help = "Optional: Maximum training window size")

    # Model
    parser.add_argument('--type_model', type = str, help = "Model Type [MLP, CNN, LSTM, GNU, TCNN, Transformer]")

    # Hyperparameters Model
    parser.add_argument('--lr', type = float, help = "Learning Rate [float]")
    parser.add_argument('--drop', type = float, help = "Dropout Model [float]")
    parser.add_argument('--patience', type = float, help = "Patience of the Error in the model [float]")
    parser.add_argument('--optimizer_type', type = str, help = "Optimizer Type [Adam, SGD]")
    parser.add_argument('--schler', type = str, help = "Scheduler Type [Reduce, Cosine, CosineRW]")

    # LSTM
    parser.add_argument('--num_layer_lstm', type = int, help = "Number Layers LSTM [int]")
    parser.add_argument('--hidden_neurons_lstm', type = int, help = "Number od Neurons in LSTM [int]")

    # GRU
    parser.add_argument('--num_layer_gru', type = int, help = "Number Layers GRU [int]")
    parser.add_argument('--hidden_neurons_gru', type = int, help = "Number od Neurons in GRU [int]")

    # CNN & TCNN
    parser.add_argument('--kernel_size', type = int, help = "Kernel CNN/TCNN Layers [int]")

    # TCNN
    parser.add_argument('--num_channel_list_tcnn', type = int, nargs = '+', help = "Number of Channels in TCNN [space-separated values]")

    # Transformer
    parser.add_argument('--d_model_transformer', type = int, help = "D Model in Transformer [int]")
    parser.add_argument('--nhad_transformer', type = int, help = "Number of Heads in Transformer [int]")
    parser.add_argument('--num_encoder_layers_transformer', type = int, help = "Number of Encoder Layers in Transformer [int]")
    parser.add_argument('--dim_feedforward_transforme', type = int, help = "Dimension Feedforward in Transformer [int]")

    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}

    return overrides

