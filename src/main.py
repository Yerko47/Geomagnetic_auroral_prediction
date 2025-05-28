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

    #* 2. Initial Division: Development and Final Testing (Hold-Out)
    # --------------------------------------------------------------
    development_df, test_df = create_set_prediction(df_scaler, cfg)
    del df_scaler

    auroral_index_target = cfg['data']['auroral_index'].strip()
    model_type_config = cfg['nn']['type_model'].strip().upper()
    
    #* 3. Cross Validation
    # --------------------
    cv_results_per_delay = []

    for delay in cfg['constant']['delay_length']:
        print(f"\n{'='*4} Cross Validation for Delay: {delay} {'='*4}")

        fold_count = 0

        best_fold_delay = {
            'best_val_metric': float('inf'),
            'best_model_path': None,
            'best_fold_id': None,
            'best_model_state_dict': None
        }
        
        batch_train_size = cfg['nn'].get('batch_train', 2080)
        batch_val_size = cfg['nn'].get('batch_val', 2080)

        for train_indices, val_indices in cross_validation(development_df, cfg):
            fold_count += 1
            fold_id_str = f"delay{delay}_fold{fold_count}"
            print(f"\n--- Processing Fold {fold_count}/{cfg.get('cv', {}).get('n_splits', 5)} (ID: {fold_id_str}) ---\n")

            fold_train_df = development_df.iloc[train_indices]
            fold_val_df = development_df.iloc[val_indices]
            
            # Apply delay and create DataLoaders for the current fold
            train_solar_fold, train_index_fold = time_delay(fold_train_df, cfg, delay, 'train')
            val_solar_fold, val_index_fold = time_delay(fold_val_df, cfg, delay, 'val')

            train_torch_fold = DataTorch(train_solar_fold, train_index_fold, device)
            val_torch_fold = DataTorch(val_solar_fold, val_index_fold, device)
        
            train_loader_fold = DataLoader(train_torch_fold, batch_size = batch_train_size, shuffle = True)
            val_loader_fold = DataLoader(val_torch_fold, batch_size = batch_val_size, shuffle = False)

            # Instantiate and train the model for this fold (new model each time)
            model_fold = type_nn(cfg, x_train_shape = train_solar_fold.shape, delay = delay, device = device)
            
            criterion_fold = nn.MSELoss()

            trained_model_fold, epoch_metrics_df_fold, best_val_loss_fold, path_save_fold  = train_val_model(
                model = model_fold, criterion = criterion_fold, 
                train_loader = train_loader_fold, val_loader = val_loader_fold,
                config = cfg, paths = project_paths, 
                delay = delay, 
                device = device, seed = 42 + fold_count, 
                fold_identifier = str(fold_count) 
            )

            # Update the best model for this delay if the actual fold is better
            if best_val_loss_fold < best_fold_delay['best_val_metric']:
                best_fold_delay({
                    'best_val_metric': best_val_loss_fold,
                    'best_model_path': path_save_fold,
                    'best_fold_id': fold_count,
                    'best_model_state_dict': deepcopy(trained_model_fold.state_dict())
                })
            
            metrics_plot(metrics_df = epoch_metrics_df_fold, config = cfg, paths = project_paths, plot_title_suffix = f"Fold{fold_count}_Delay{delay}")
           
        #* 4. Testing the Best Model for the Current Delay
        # ------------------------------------------------
        test_solar, test_index, test_epoch = time_delay(test_df.copy(), cfg, delay, 'test')

        batch_test_size = cfg['nn'].get('batch_test', 1040)
        test_torch = DataTorch(test_solar, test_index, device)
        test_loader = DataLoader(test_torch, batch_size = batch_test_size, shuffle = False)

        model_test = type_nn(cfg, x_train_shape = test_solar.shape, delay = delay, device = device)
        model_test.load_state_dict(best_fold_delay['best_model_state_dict'])

        criterion = nn.MSELoss()

        result_df, test_metrics_df = testing_model(
            model = model_test, criterion = criterion,
            test_loader = test_loader, config = cfg, 
            paths = project_paths, delay = delay, 
            test_epoch = test_epoch, device = device
        )

        

        



if __name__ == "__main__":
    main()
    