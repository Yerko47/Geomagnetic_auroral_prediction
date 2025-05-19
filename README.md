project_root/
├── config/
│   └── config.yaml             # Editable parameters
├── data/
│   ├── raw/                    # Raw data
│   └── processed/              # Processed data
├── models/
│   └── results_csv/            # CSV with results and metrics
├── plots/                      # Generated plots
├── src/                        # Source code
│   ├── __init__.py
│   ├── main.py                 # Entry point that executes the pipeline
│   ├── cli.py                  # CLI to override configuration
│   ├── config_loader.py        # Loads YAML config and applies overrides
│   ├── variables.py            # Variables and paths from final config
│   ├── data_processing.py      # Data preprocessing
│   ├── model_utils.py          # Utility functions for models
│   ├── model_training.py       # Model training and testing
│   └── models.py               # Definition of networks (ANN, CNN, LSTM, Transformer, GRU)
├── tests/                      # Unit tests
├── docs/                       # Project documentation
└── notebooks/                  # Interactive exploration and visualization
