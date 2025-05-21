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
        
            batch_train_size = cfg['nn'].get('batch_train', 2080)
            batch_val_size = cfg['nn'].get('batch_val', 2080)

            train_loader_fold = DataLoader(train_torch_fold, batch_size = batch_train_size, shuffle = True)
            val_loader_fold = DataLoader(val_torch_fold, batch_size = batch_val_size, shuffle = False)

            model_fold = type_nn(cfg, x_train_shape = train_solar_fold.shape, delay = delay, device = device)
            
            criterion_fold = nn.MSELoss()

            _, epoch_metrics_df_fold = train_val_model(
                model = model_fold, criterion = criterion_fold, 
                train_loader = train_loader_fold, val_loader = val_loader_fold,
                config = cfg, paths = project_paths, 
                delay = delay, 
                device = device, seed = 42 + fold_count, 
                fold_identifier = str(fold_count) 
            )
            break
        

if __name__ == "__main__":
    main()