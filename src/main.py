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
    # --------------------------------
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
    
    # Processing and plotting flags from config
    processOMNI = cfg.get('constant', False)
    processPLOT = cfg.get('plot', False)


    #* 1. LOAD AND PROCESS DATA
    # --------------------------
    df = dataset(cfg, project_paths, processOMNI)
    df_storm = storm_selection(df, project_paths)
    df_scaler = scaler_df(df_storm, cfg)

    del df, df_storm

    development_df, test_df = create_set_prediction(df_scaler, cfg)
    del df_scaler

    auroral_index_target = cfg['data']['auroral_index'].strip()
    model_type_config = cfg['nn']['type_model'].strip().upper()

    print(development_df.shape, test_df.shape)
        
