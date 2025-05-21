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

    cv_results_per_delay = []

    for delay in cfg['constant']['delay_length']:
        print(f"\n===== Cross Validation for Delay: {delay} =====")

        fold_validation_metrics = []
        fold_count = 0

        for train_indices, val_indices in cross_validation(development_df, cfg):
            fold_count += 1
            fold_id_str = f"delay{delay}_fold{fold_count}"
            print(f"\n--- Processing Fold {fold_count}/{cfg.get('cv', {}).get('n_splits', 5)} (ID: {fold_id_str}) ---")

            fold_train_df = development_df.iloc[train_indices]
            fold_val_df = development_df.iloc[val_indices]

            train_solar_fold, train_index_fold = time_delay(fold_train_df, cfg, delay, 'train')
            val_solar_fold, val_index_fold = time_delay(fold_val_df, cfg, delay, 'val')

            print(f"Train shape: {train_solar_fold.shape}, Val shape: {val_solar_fold.shape}")

            train_torch_fold = DataTorch(train_solar_fold, train_index_fold, device)
            val_torch_fold = DataTorch(val_solar_fold, val_index_fold, device)
        

if __name__ == "__main__":
    main()