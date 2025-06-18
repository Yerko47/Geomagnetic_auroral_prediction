import os
import cdflib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Tuple, Any, Union, Iterable

#* READ CDF
def cdf_read(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads data from a CDF (Common Data Format) file into a pandas DataFrame.

    This function opens a specified CDF file, extracts all variables, converts the 'Epoch' variable to a readable datetime format, and renames specific columns ('E' to 'E_Field', 'F' to 'B_Total') for clarity.

    Args:
        file_path (Union[str, Path]): 
            The path to the CDF file
    
    Returns:
        cdf_df (pd.Dataframe):
            A DataFrame containing the data from the CDF file. Returns an empty DataFrame if the file cannot be processed.
    """

    try:
        cdf = cdflib.CDF(file_path)     # Open the CDF file
    except Exception as e:
        print(f'Error opening or reading CDF file {file_path}: {e}')
    
    cdf_dict = {}       # Initialize a dictionary to store data from CDF variables
    info = cdf.cdf_info()       # Get information about the CDF file, including variables

    # Iterate over all zVariables (variables that depend on record variance)
    for key in info.zVariables:
        cdf_dict[key] = cdf[key][...]       #Extract data for each variable
        
    
    cdf_df = pd.DataFrame(cdf_dict)

    # Convert 'Epoch' column from CDF epoch format to pandas datetime objects
    if 'Epoch' in cdf_df.columns:
        try:
            cdf_df['Epoch'] = pd.to_datetime(cdflib.cdfepoch.encode(cdf_df['Epoch'].values))
        except Exception as e:
            print(f"Warning: Could not convert 'Epoch' column for {file_path}. Error: {e}")

    # Rename columns for consistency and clarity if they exist
    column_renames = {'E': 'E_Field', 'F': 'B_Total'}
    cdf_df.rename(columns=column_renames, inplace=True)

    return cdf_df


#* DATA CLEANING
def bad_data(cdf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by identifying and handling potential bad data points.

    For each numeric column (excluding 'Epoch' or datetime columns), it calculates a threshold based on the maximum value (rounded down to two decimal places). Values greater than or equal to this threshold are replaced with NaN. These NaN values are then interpolated linearly, and any remaining NaNs at the beginning or end are backfilled (up to a limit of 3).

    Args:
        cdf_df (pd.DataFrame):
            The input DataFrame with potentially erroneous data.

    Returns:
        processed_df (pd.DataFrame):
            The processed DataFrame with erroneous values handled.
    """

    if cdf_df.empty:        # Check if the dataframe is empty
        return cdf_df

    processed_df = cdf_df.copy()

    for col in processed_df.columns:
        if col == 'Epoch': continue

        # Calculate the maximun value, floored to two decimal places.
        max_value_threshold = np.floor(processed_df[col].max() * 100) / 100

        #Identify values that are greater than or equal to this threshold as "bad"
        processed_df.loc[processed_df[col] >= max_value_threshold, col] = np.nan

        # Interpolate NaN values using linear method
        processed_df[col] = processed_df[col].interpolate(method = 'linear')

        # Backfill any remaining NaN values, limiting to 3 to avoid excessive propagation
        processed_df[col] = processed_df[col].bfill(limit = 3)

        # Forward fill any still remaining NaN values (e.g., if all initial values were NaN)
        processed_df[col] = processed_df[col].ffill(limit=3)
    
    return processed_df


#* DATASET BUILDING
def dataset(config: Dict[str, Any], paths: Dict[str, Path], processOMNI: bool = True) -> pd.DataFrame:
    """
    Builds the OMNI dataset from CDF files over a specified year range.

    If `processOMNI` is True and the processed data doesn't exist as a feather file, it reads raw OMNI CDF files month by month, concatenates them, cleans the data using `bad_data`, drops irrelevant columns, and saves the result as a feather file. If the feather file already exists, or if `processOMNI` is False (and file exists), it loads the data directly from the feather file.

    Args:
        config (Dict[str, Any)]:
            Configuration dictionary containing parameters like 'in_year', 'out_year', and 'omni_data_base_path'.
        paths (Dict[str, Path]):
            Dictionary of project paths, expecting 'raw_file' (directory for saving/loading feather file).
        processOMNI (bool, optional):
            Whether to re-process raw OMNI CDF files. Defaults to True.

    Returns:
        df_omni (pd.DataFrame):
            The cleaned and processed OMNI dataset.
    """

    # Extract parameters from config
    in_year = config['constant']['in_year']
    out_year = config['constant']['out_year']

    omni_path = f'/data/omni/hro_1min/'
    omni_path = Path(omni_path)

    # Define start and end timestamps for data retrieval
    # Adjusted end_time to cover the full last year requested
    start_time = pd.Timestamp(f'{in_year}-01-01')
    end_time = pd.Timestamp(f'{out_year}-12-31')        # Cover the whole last year

    # Path for the processed feather file
    save_feather_path = paths['raw_file'] / f'omni_data_{in_year}_to{out_year}.feather'

    # Columns to drop from the OMNI dataset (specific to OMNI HRO 1-min data)
    columns_to_drop = [
        'YR', 'Day', 'HR', 'Minute', 'PLS', 'PLS_PTS',
        'percent_interp', 'Timeshift', 'RMS_Timeshift', 'RMS_phase',
        'Time_btwn_obs', 'RMS_SD_B', 'RMS_SD_fld_vec', 'Mach_num',
        'Mgs_mach_num', 'ASY_D', 'ASY_H', 'PC_N_INDEX',
        'x', 'y', 'z' # Assuming GSM/GSE coordinates are preferred if present
    ]

    if processOMNI and not save_feather_path.exists():
        print(f"\nProcessing OMNI data from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}\n" 
              f"{':' * 50} \n")
        
        # Generate a date range for fetching monthly files
        date_array = pd.date_range(start = start_time, end = end_time, freq = 'MS')     # Generate monthly dates (freq = 'MS' --> start of every month)

        data_frame_list = []        # List to hold monthly DataFrames

        for date in date_array:
            # Construct the path to the monthly CDF file
            # File naming convention: omni_hro_1min_YYYYMMDD_v01.cdf
            # The OMNI archive typically has one file per month, named with the first day of the month.
            file_name = f"omni_hro_1min_{date.strftime('%Y%m01')}_v01.cdf"

            cdf_file_path = omni_path / str(date.year) / file_name

            if cdf_file_path.exists():
                print(f'Loading file: {cdf_file_path}')
                cdf_df = cdf_read(cdf_file_path)        # Reat the CDF file
                if not cdf_df.empty:
                    data_frame_list.append(cdf_df)
            else:
                print(f'Warning: File not found {cdf_file_path}')

        if not data_frame_list:
            print("No data files were successfully loaded. Returning an empty DataFrame.")
            return pd.DataFrame()
            
        # Concatenate all monthly DataFrames
        df_omni = pd.concat(data_frame_list, axis = 0, ignore_index = True)
        
        
        # Drop specified irrelevant columns, handle missing columns gracefully
        existing_cols_to_drop = [col for col in columns_to_drop if col in df_omni.columns]
        df_omni.drop(columns = existing_cols_to_drop, inplace = True)

        # Clean the data (interpolate NaNs, handle outliers)
        df_omni = bad_data(df_omni)


        # Save the processed DataFrame to a feather file
        # Reset index if Epoch was made index but is also a column
        df_omni.reset_index(drop = True, inplace = True) 
        df_omni.to_feather(save_feather_path)
        print(f'Successfully processed and saved {len(df_omni)} data points to {save_feather_path}')

        return df_omni

    elif save_feather_path.exists():
        print(f'Loading processed OMNI data from {save_feather_path}')
        df_omni = pd.read_feather(save_feather_path)
        return df_omni
    
    else:
        # Case: processOMNI is False but file does not exist
        raise FileNotFoundError(
            f"The processed OMNI data file {save_feather_path} does not exist, "
            "and processOMNI is set to False. Set processOMNI to True to generate the file."
        )