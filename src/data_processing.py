import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Tuple, Any, Union, Iterable

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit

import torch
from torch.utils.data import Dataset


#* STORM SELECTION
def storm_selection(df: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    """
    Extracts time windows around storm events from the dataset.

    Reads a list of storm event timestamps from 'storm_list.csv' located in the directory specified by `paths['processed_file']`. For each storm, it extracts a 48-hour window (24 hours before and 24 hours after the storm's recorded onset).

    Args:
        df (pd.DataFrame):
            The input time series DataFrame, expected to have an 'Epoch' column.
        paths (Dict[str, Path]): 
            Dictionary of project paths, expecting 'processed_file' (directory containing 'storm_list.csv').

    Returns:
        df_storm (pd.DataFrame):
            A DataFrame containing continuous 48-hour segments for each identified storm event. Returns an empty DataFrame if no storms are processed or 'storm_list.csv' is not found.
    """

    # Path to the storm list CSV file
    storm_list_path = paths['processed_file'] / 'storm_list.csv'

    if not storm_list_path.exists():
        print(f"Warning: Storm list file not found at {storm_list_path}. Returning original DataFrame.")
        return df

    # Load the list of storm event timestamps
    storm_list_df = pd.read_csv(
        storm_list_path,
        header = None,
        names = ['Epoch'],
        parse_dates = ['Epoch']
    )

    if 'Epoch' not in df.columns:
        print("Error: 'Epoch' column not found in the input DataFrame for storm_selection.")
        return pd.DataFrame()
    
    # Ensure the main DataFrame's 'Epoch' column is in datetime format and set as index
    df['Epoch'] = pd.to_datetime(df['Epoch'])
    df_indexed = df.set_index('Epoch')      # Work with a DatetimeIndex for efficient slicing

    selected_storm_segments = [] # List to store DataFrames for each storm window

    # Iterate over each storm event timestamp
    for storm_time in storm_list_df['Epoch']:
        #Define the start and end to the 48-hour window arround the storm
        start_window = storm_time - pd.Timedelta(hours = 36)
        end_window = storm_time + pd.Timedelta(hours = 36)

        #Extract the data for this window from the main DataFrame
        storm_segment = df_indexed.loc[start_window:end_window].copy()
        
        # Append the extracted segment to the list
        if not storm_segment.empty:
            selected_storm_segments.append(storm_segment)
    
    if not selected_storm_segments:
        print("No storm segments were extracted.")
        return pd.DataFrame()

    # Concatenate all storm segments into a single DataFrame
    df_storm = pd.concat(selected_storm_segments, axis=0)

    # Reset the index to make 'Epoch' a regular column again
    return df_storm.reset_index()


#* SCALER DATASET
def scaler_df(df_storm: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Scales numerical features in the DataFrame and handles AL_INDEX negation.

    Extracts 'Epoch', solar parameters (features), and auroral parameters (targets) based on `config`. Solar parameters are scaled using the method specified in `config['data']['scaler_type']`. If 'AL_INDEX' is among the auroral parameters, its values are negated (multiplied by -1) to make them positive, as AL is typically negative.

    Args:
        df_storm (pd.DataFrame): 
            The input DataFrame, expected to contain 'Epoch', solar parameters, and auroral parameters.
        config (Dict[str, Any]): 
            Configuration dictionary containing:
            - `config['data']['scaler_type']`: Type of scaler ('robust', 'standard', 'minmax').
            - `config['constant']['omni_param']`: List of solar parameter column names.
            - `config['constant']['auroral_param']`: List of auroral parameter column names.
    
    Returns: 
        df_processed (pd.DataFrame):
            The DataFrame with solar parameters scaled and AL_INDEX potentially negated. Returns an empty DataFrame if essential columns are missing.
    """

    if df_storm.empty:
        return pd.DataFrame()
    
    # Extract column names from config
    omni_param_cols = config['constant']['omni_param']
    auroral_param_cols = config['constant']['auroral_param']
    scaler_type = config['data']['scaler_type'].strip()         # Remove potential spaces

    missing_omni = [col for col in omni_param_cols if col not in df_storm.columns]
    missing_auroral = [col for col in auroral_param_cols if col not in df_storm.columns]

    if missing_omni:
        print(f"Warning: Missing OMNI parameter columns in scaler_df: {missing_omni}. These will be skipped.")
        # Filter out missing columns to avoid errors
        omni_param_cols = [col for col in omni_param_cols if col in df_storm.columns]
        if not omni_param_cols:
            print("Error: No OMNI parameters left to scale.")
            return pd.DataFrame()
            
    if missing_auroral:
        print(f"Warning: Missing Auroral parameter columns in scaler_df: {missing_auroral}. These will be skipped.")
        auroral_param_cols = [col for col in auroral_param_cols if col in df_storm.columns]
        if not auroral_param_cols:
             print("Warning: No Auroral parameters found.")         # Can still proceed if only scaling solar

    # Separate parts of the DataFrame
    df_epoch = df_storm[['Epoch']].copy()
    df_solar = df_storm[omni_param_cols].copy()
    df_index = df_storm[auroral_param_cols].copy()

    # Negate 'AL_INDEX' if it exists in the auroral parameters, to make its values positive
    # AL_INDEX is typically negative, representing westward electrojet strength.
    if 'AL_INDEX' in df_index.columns:
        df_index['AL_INDEX'] = -1 * df_index['AL_INDEX']

    # Initialize the scaler based on the type specified in config
    match scaler_type:
        case 'robust': scaler = RobustScaler()
        case 'standard': scaler = StandardScaler()
        case 'minmax': scaler = MinMaxScaler()
        case _: raise ValueError(f"Invalid scaler_type: {scaler_type}. Choose from 'robust', 'standard', 'minmax'.")

    df_solar_scaled = scaler.fit_transform(df_solar)

    # Convert the scaled numpy array back to a DataFrame with original column names and index
    df_solar_scaled = pd.DataFrame(df_solar_scaled, columns=df_solar.columns, index=df_solar.index)

    # Concatenate the 'Epoch' column, scaled solar parameters, and (potentially modified) auroral indices
    df_processed = pd.concat([df_epoch, df_solar_scaled, df_index], axis=1)

    return df_processed


#* SET PREDICTION
def create_set_prediction(df_processed: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training, validation, and test sets.

    The splitting strategy is determined by `config['data']['set_split']`:
        - 'organized': Performs a sequential split based on time.
        - 'random': Performs a stratified random split (though stratification isn't explicitly implemented here, it uses `train_test_split` which can be extended for stratification if a `stratify` column is provided).Shuffle is True for train/val split in 'random' mode.

    The sizes of the test and validation sets are determined by `config['data']['test_size']` and `config['data']['val_size']` respectively.

    Args:
        df_processed (pd.DataFrame): 
            The input DataFrame to be split.
        config (Dict[str, Any]): 
            Configuration dictionary containing:
                - `config['data']['set_split']`: Splitting strategy ('organized' or 'random').
                - `config['data']['test_size']`: Proportion for the test set.
                - `config['data']['val_size']`: Proportion for the validation set (from the remainder after test split).

    Returns:
        Tuple ([pd.DataFrame, pd.DataFrame, pd.DataFrame]): 
            DataFrames for training, validation, and testing.
    """                                                          

    if df_processed.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    set_split_strategy = config['data']['set_split'].strip()        
    test_proportion = config['data']['test_size']                  

    if set_split_strategy == 'organized':
        n_total = len(df_processed)

        test_start_index = int(n_total * (1 - test_proportion))

        development_df = df_processed.iloc[:test_start_index].copy()
        test_df = df_processed.iloc[test_start_index:].copy()

    elif set_split_strategy == 'random':
        development_df, test_df = train_test_split(
            df_processed,
            test_size = test_proportion,
            shuffle = False
        )

    else:
        raise ValueError(f"Invalid set_split strategy: {set_split_strategy}. Choose 'organized' or 'random'.")
    
    # Reset indices for all resulting DataFrames
    development_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    print(f"Data split ({set_split_strategy}): Development = {len(development_df)}, Test = {len(test_df)}")
    return development_df, test_df


#* CROSS VALIDATION
def cross_validation(development_df: pd.DataFrame, config: Dict[str, Any]) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates the indexes for the training and validation folds for TimeSeries Cross-Validation.

    Args:
        development_df (pd.DataFrame): 
            The development dataset.
        config (Dict[str, Any]): 
            Configuration dictionary that may contain:
                - config['cv']['n_splits']: Number of folds for TimeSeriesSplit (default: 5).
                - config['cv']['gap']: Number of samples to omit between train and test set in each fold (default: 0).
                - config['cv']['max_train_size']: Maximum size for each training set (default: None).

    """

    n_split = config['cv']['n_split']
    gap = config['cv']['gap']
    max_train_size = len(development_df)//2 if config['cv']['max_train_size'] == 0 else config['cv']['max_train_size']

    n_split_new = max(2, n_split) if len(development_df) > n_split else 1

    tscv = TimeSeriesSplit(
        n_splits = n_split_new,
        gap = gap,
        max_train_size = max_train_size
    )

    print(f"Generating {n_split_new} folds for Cross Validation (TimeSeriesSplit)...")

    for train_index, val_index in tscv.split(development_df):
        if len(train_index) == 0 or len(val_index) == 0:
            print(f"Warning: A fold was generated with an empty training or validation set. Skipping this fold.")
            continue
        yield train_index, val_index
            

#* TIME DELAY
def time_delay(df: pd.DataFrame, config: Dict[str, Any], delay: int, dataset_group: str) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, pd.Series]]:
    """
    Prepares time-series data with lag features (time delay) for input to models.

    The shape of the output solar wind features depends on `config['nn']['type_model']`:
    - 'ANN': Features are flattened [num_samples, num_features * delay_steps].
    - 'LSTM, GRU, Transformer': Features are shaped as sequences [num_samples, delay_steps, num_features].
    - 'CNN, TCNN': Features are shaped as sequences [num_samples, num_features, delay_steps] (channels first for Conv1D).

    Args:
        df (pd.DataFrame): 
            Input DataFrame containing 'Epoch', solar parameters, and the target auroral index.
        config (Dict[str, Any]): 
            Configuration dictionary containing:
            - `config['constant']['omni_param']`: List of solar parameter column names.
            - `config['data']['auroral_index']`: Name of the target auroral index column.
            - `config['nn']['type_model']`: Type of neural network model ('ANN', 'LSTM', 'CNN').
        delay_steps (int): 
            Number of time steps to include in the lag features (time window).
        dataset_group (str): 
            Specifies the dataset type ('train', 'val', 'test'). If 'test', 'Epoch' data is also returned.

    Returns:
        Union ([Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, pd.Series]]):
            - np_solar (np.ndarray): Solar wind features with time delay applied.
            - np_index (np.ndarray): Target auroral index values.
            - df_epoch (pd.Series, optional): Series of 'Epoch' timestamps (returned only if `dataset_group` is 'test').
    """
       
    omni_param = config['constant']['omni_param']
    auroral_index = config['data']['auroral_index'].upper()
    type_model = config['nn']['type_model'].strip()
    
    if not all(col in df.columns for col in omni_param):
        raise ValueError("Missing one or more OMNI parameter columns in the time_delay input DataFrame.")
    if not auroral_index in df.columns:
        raise ValueError(f"Target auroral index '{auroral_index}' not found in time_delay input DataFrame.")
    if 'Epoch' not in df.columns and dataset_group == 'test':
        raise ValueError("'Epoch' column missing, required for test group in time_delay.")
    

    df_solar = df[omni_param].copy()
    df_index = df[auroral_index].copy()
    np_index = df_index.to_numpy()
    

    # Store Epoch data if processing the test set
    if dataset_group == 'test':
        df_epoch = df['Epoch'].copy()

    # For ANN, create lag features by shifting and concatenating columns
    if type_model == 'ANN':
        lagged_columns = []

        for lag in range(delay):
            shifted_features = df_solar.shift(lag + 1)
            shifted_features.columns = [f"{col}_{lag}" for col in df_solar.columns]
            lagged_columns.append(shifted_features)
        
        df_solar_with_lags = pd.concat(lagged_columns, axis = 1)
        df_solar_with_lags.dropna(inplace = True)
        np_solar = df_solar_with_lags.values.astype(np.float32)

        np_index = np_index[delay:]
        if dataset_group == 'test':
            df_epoch = df_epoch.iloc[delay:].reset_index(drop = True)
    
    # Models expecting [batch_size, sequence_length, num_features]
    elif type_model in ['LSTM', 'GRU', 'TRANSFORMER']:
        sequences = []
        for i in range(len(df_solar) - delay + 1):
            sequence = df_solar.iloc[i : i + delay].values
            sequences.append(sequence)
        
        np_solar = np.array(sequences, dtype = np.float32)
        np_index = np_index[delay - 1:]
        if dataset_group == 'test' and not df_epoch.empty:
            df_epoch = df_epoch.iloc[delay - 1:].reset_index(drop = True)

    # Models expecting [batch_size, num_features, sequence_length]
    elif type_model in ['CNN', 'TCNN']:
        sequences = []
        for i in range(len(df_solar) - delay + 1):
            sequence = df_solar.iloc[i : i + delay].values
            sequences.append(sequence)

        np_solar = np.array(sequences, dtype = np.float32)
        np_solar = np.transpose(np_solar, (0, 2, 1)) 

        np_index = np_index[delay - 1:]
        if dataset_group == 'test' and not df_epoch.empty:
            df_epoch = df_epoch.iloc[delay - 1:].reset_index(drop = True)
    
    else:
        raise ValueError(f"Invalid type_model: {type_model}. Choose from 'ANN', 'LSTM', 'GRU', 'TRANSFORMER', 'CNN', 'TCNN'.")
    
    # Return results
    if dataset_group == 'test':
        return np_solar, np_index, df_epoch
    else:
        return np_solar, np_index
    

#* DATA TORCH
class DataTorch(Dataset):
    """
    A PyTorch Dataset class for handling solar wind and auroral index data.

    Args:
        solar_wind_features (np.ndarray): 
            NumPy array of solar wind parameters (features).
        auroral_index_target (np.ndarray): 
            NumPy array of auroral index values (target).
        device (Union[str, torch.device]): 
            The target device for tensor placement.
    """

    def __init__(self, np_solar: np.ndarray, np_index: np.ndarray, device: Union[str, torch.device]):
        self.device = device
        self.x_data = torch.tensor(np_solar, dtype = torch.float32).to(self.device)
        self.y_data = torch.tensor(np_index, dtype = torch.float32).unsqueeze(1).to(self.device)
    
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.x_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample (features and target) from the dataset at the given index.
        """
        return self.x_data[idx], self.y_data[idx]
        
        
