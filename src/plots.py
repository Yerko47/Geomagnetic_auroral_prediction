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
        