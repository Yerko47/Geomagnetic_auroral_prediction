import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from config_loader import *
from cli import *
from paths import *
from data_processing import *
from model_training import *

from plots import *

def main():
    """
    
    """
    #* 0. INITIALIZE PROJECT STRUCTURE
    # -------------------------------
    overrides = parse_args()
    cfg = config_load(overrides = overrides)

    # Seed
    set_seed(42)

    # Project Paths
    project_paths = structure_project(project_root = cfg.get('project_root', None))

    # Determine the device (CUDA or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    


