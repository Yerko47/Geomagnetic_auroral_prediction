import yaml
from pathlib import Path

#* READ YALM FILE
def config_load(config_path: str = None, overrides: dict = None) -> dict:
    """
    Loads the configuration file and applies overrides from the CLI.

    Args:
        config_path (str):
            Path to the config.yaml file
        overrides (dict):
            Dictionary with keys and values to override the configuration
    
    Returns:
        cdg (dict):
            Final merged configuration

    """
    
    path = Path(config_path or Path(__file__).parent.parent / 'config' / 'config.yaml' )
    
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides according to the corresponding path in the dictionary
    for key, value in overrides.item():
        # Data
        if key in ['auroral_index', 'scaler_type', 'set_split', 'test_size']:
            cfg['data'][key] = value

        if key in ['n_split', 'gap', 'max_train_size']:
            cfg['cv'][key] = value

        # Model and hyperparameters
        elif key in ['type_model', 'lr', 'drop', 'patience', 'optimizer_type', 'schler', 'kernel_cnn', 'num_layer_lstm', 'hidden_neurons_lstm', 'num_layer_gru', 'hidden_neurons_gru', 'num_chanel_list_tcnn', 'kernel_size_tcnn', 'd_model_transformer', 'nhead_transformer', 'num_encoder_layers_transformer', 'dim_feedforward_transformer']:
            cfg["nn"][key] = value
        
        
    
    return cfg