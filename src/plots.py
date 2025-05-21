import numpy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.dates import mdates
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
    
    ax.tick_params(axis = 'both', length = 7, width = 2, colors = 'black', grid_color = 'black', grid_alpha = 0.4, wich = 'major', labelsize = ticksize)
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
    """
    storm_list_file = paths['processed_file'] / 'storm_list.csv'
    