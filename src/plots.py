import numpy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
#from matplotlib.dates import mdates
from pathlib import Path
from typing import Dict, List, Any, Union

import imageio.v2 as imageio
from io import BytesIO

#* SETUP AXIS STYLE
def setup_axis_style(ax: plt.Axes, xlabel: str = None, ylabel: str = None, xlabelsize: int = 15, ylabelsize: int = 15, ticksize: int = 15) -> None:
    """
    Configure axis labels, tick parameters, and grid style for a matplotlib Axes object.

    Args:
        ax (plt.Axes):
            The matplotlib Axes object to configure.
        xlabel (str, optional):
            Label for the x-axis. Defaults to None.
        ylabel (str, optional):
            Label for the y-axis. Defaults to None.
        xlabelsize (int, optional):
            Font size for the x-axis label. Defaults to 15.
        ylabelsize (int, optional):
            Font size for the y-axis label. Defaults to 15.
        ticksize (int, optional):
            Font size for axis tick labels. Defaults to 15.
    """

    if xlabel:
        ax.set_xlabel(xlabel, fontsize = xlabelsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize = ylabelsize)
    
    ax.tick_params(axis = 'both', length = 7, width = 2, color = 'black', grid_color = 'black', grid_alpha = 0.4, which = 'major', labelsize = ticksize)
    ax.grid(True)

    if ylabel:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText = True))
        ax.yaxis.get_major_formatter().set_powerlimits((-3, 4))
        

#* FORMAT PARAMETERS LABEL
def format_params_label(param: str) -> str:
    """
    Format a parameter name into a more readable and LaTeX-friendly label for plots.

    Args:
        param (str):
            The parameter name to format.
    Returns:
        param (str): 
            The formatted label string for use in plot axes or legends.
    """

    if 'INDEX' in param:
        return param.replace('_INDEX', ' ') + r'[nT]'
    elif 'beta' in param:
        return r'$\beta$'
    elif 'B' in param:
        if 'Total' in param:
            return param.replace('_Total', 'T ') + r'[nT]'
        else:
            return param.replace('_', ' ') + r'[nT]'
    elif 'proton' in param:
        return r'$\rho ' + r'[#N/cm$^3$]'
    elif 'V' in param:
        return param + r'[km/s]'
    elif 'flow_speed' in param:
        return param.replace('_', ' ') + r' [km/s]'
    elif param == 'T' and 'B' not in param:
        return param + r' [K]'
    elif 'E_Field' in param:
        return 'E ' + r'[mV/m]'
    elif 'Pressure' in param:
        return 'P ' + r'[nPa]'
    elif 'IMF' in param:
        return param + r' [nT]'
    return param


#* MAPPING COLUMNS
def get_column_mapping(df_columns: List[str]) -> Dict[str, str]:
    """
    Generates a dictionary to rename DataFrame columns into more readable or scientific formats.
    Args:
        df_columns (List[str]): 
            List of column names from a DataFrame.
    Returns:
        columnn_mapping (dict): 
            A dictionary mapping original column names to formatted labels.
    """
    column_mapping = {}
    for col_name in df_columns:
        new_name = col_name 
        if 'INDEX' in col_name:
            new_name = col_name.replace('_INDEX', '')
        if 'GSM' in col_name:
            new_name = col_name.replace('_', ' ')
        if 'GSE' in col_name:
            new_name = col_name.replace('_', ' ')
        if 'Total' in col_name: 
            new_name = col_name.replace('_Total', 'T')
        if 'proton' in col_name: 
            new_name = r'$\rho$'
        if 'Pressure' in col_name:
            new_name = 'P'
        if 'E_Field' in col_name:
            new_name = 'E'
        if 'Beta' in col_name:
            new_name = r'$\beta$'
        if 'flow_speed' in col_name:
            new_name = 'Flow Speed'
        column_mapping[col_name] = new_name
    return column_mapping


#* LOAD STORM DATA
def load_storm_data(paths: Dict[str, Path], min_date_str: str = None) -> pd.DataFrame:
    """
    Loads storm event data from 'storm_list.csv' and filters by a minimum date.

    Args:
        paths (Dict[str, Path]): 
            Dictionary of project paths, expecting 'processed_file'.
        min_date_str (str, optional): 
            Minimum date (YYYY-MM-DD) for filtering.
    Returns:
        storm_df (pd.DataFrame): 
            DataFrame with storm 'Epoch' timestamps.
    """
    storm_list_file = paths['processed_file'] / 'storm_list.csv'
    if not storm_list_file.exists():
        ValueError(f"Storm list file not found at {storm_list_file}")

    storm_df = pd.read_csv(storm_list_file, header = None, names = ['Epoch'])
    storm_df['Epoch'] = pd.to_datetime(storm_df['Epoch'])

    if min_date_str:
        min_date = pd.to_datetime(min_date_str)
        storm_df = storm_df[storm_df['Epoch'] >= min_date]
    
    return storm_df
    
#* METRICS PLOT (TRAINING/VALIDATION CRUVES)
def metrics_plot(metrics_df: pd.DataFrame, config: Dict[str, Any], paths: Dict[str, Path], plot_title_suffix: str = "") -> None:
    """
    Generates plots for training and validation metrics (RMSE, R-Score).

    Args:
        metrics_df (pd.DataFrame): 
            DataFrame with metrics history per epoch. Column names should follow patterns like 'Train Rmse_DELAY_FOLD', 'Val Rmse_DELAY_FOLD'.
        config (Dict[str, Any]): 
            Configuration dictionary.
        paths (Dict[str, Path]): 
            Project paths dictionary.
        plot_title_suffix (str): 
            Suffix to add to plot titles and filenames (e.g., fold ID, "FinalRetrained").
    """
    metric_bases = ['rmse', 'r']
    auroral_index = config['data']['auroral_index'].replace("_INDEX", " Index")
    model_type = config['nn']['type_model']

    # Extract delay from column names
    delay_str_part = ""
    first_col_parts = metrics_df.columns[0].split('_')
    if len(first_col_parts) > 1 and first_col_parts[1].isdigit():
        delay_str_part = f"Delay ({first_col_parts[1]})"

    plot_title_prefix = f"{auroral_index}-{model_type} {delay_str_part}"
    filename_prefix = f"{model_type}{delay_str_part}_{auroral_index.replace(' Index', '')}"
    if plot_title_suffix:
        plot_title_prefix += f"-{plot_title_suffix}"
        filename_prefix += f"_{plot_title_suffix}" 

    for metric in metric_bases:
        train_col = next((col for col in metrics_df.columns if metric in col and 'train' in col), None)
        val_col = next((col for col in metrics_df.columns if metric in col and 'val' in col), None)

        plt.figure(figsize = (10, 6))
        plt.title(f"{plot_title_prefix} - {metric.upper()} vs Epochs")

        if train_col:
            plt.plot(metrics_df.index + 1, metrics_df[train_col], label = 'Train', color = 'blue')
        if val_col:
            plt.plot(metrics_df.index + 1, metrics_df[train_col], label = 'Valid', color = 'red')

        setup_axis_style(plt.gca(), xlabel = 'Epoch', ylabel = metric.upper(), ticksize = 12)
        plt.legend(fontsize = 12)

        # Determine save directory based on metric type
        save_dir = None
        if 'rmse' in metric: save_dir = paths['training_rmse']
        elif 'r' in metric: save_dir = paths['training_rscore']
        else: continue

        filename = save_dir / f"{filename_prefix}_{metric.replace(' ', '')}.png"
        plt.savefig(filename)
        plt.close()
    
    print(f"Saved training/validation metric plots")
        

#* DELAY METRICS PLOT
def delay_metrics_plot(cv_performance_df: pd.DataFrame, config: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """
    Generates plots of average CV metrics vs. different delay lengths.

    Args:
        cv_performance_df (pd.DataFrame): 
            DataFrame from main.py with columns like 'delay', 'avg_cv_rmse', 'avg_cv_r_score'.
        config (Dict[str, Any]): 
            Configuration dictionary.
        paths (Dict[str, Path]): 
            Project paths dictionary.
    """
    if cv_performance_df.empty or 'delay' not in cv_performance_df.columns:
        return
    
    auroral_index = config['data']['auroral_index'].replace('_INDEX', ' Index')
    model_type = config['nn']['model_type']
    delay_values = sorted(cv_performance_df['delay'].unique())

    # Metrics to plot, assuming columns like 'arg_cv_rmse', 'avg_cv_r_score' exist
    metrics_to_plot = {
        'avg_cv_rmse': ('Average CV RMSE', paths['test_rmse']),
        'avg_cv_r': ('Average CV R', paths['test_rscore'])
    }

    for metric_col, (metric_label, save_dir) in metrics_to_plot.items():
        if metric_col not in cv_performance_df.columns:
            continue
        
        plt.figure(figsize = (10, 6))
        title = f"{metric_label} vs Delay ({auroral_index}) - {model_type}"
        plt.title(title, fontsize = 16, fontweight = 'bold')

        plt.plot(cv_performance_df['delay'], cv_performance_df[metric_col], marker = 'o', color = 'dodgerblue', linewidth = 2, linestyle = '-', label = metric_label, markersize = 8)
        plt.legend(fontsize = 12)
        setup_axis_style(plt.gca(), xlabel = 'Delay', ylabel = metric_label, ticksize = 12)

        filname_base = metric_col.replace('avg_cv_', '').replace('_', '')
        filname = save_dir / f"{filname_base}_vs_delay_{auroral_index.replace(' Index', '')}_{model_type}.png"
        plt.savefig(filname)
        plt.close()

    print(f"CV vs Delay performance plots saved.")



