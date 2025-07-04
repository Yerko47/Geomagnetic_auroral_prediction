import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from typing import Tuple, Dict, Any, Union, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import root_mean_squared_error, r2_score

#* METRICS CALCULATION
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float]:
    """
    Calculate RMSE and R2 Score Metrics
    
    Args:
        y_true (np.ndarray):
            True Values
        y_pred (np.ndarray):
            Predicted Values
    
    Returns:
        metrics (Tuple[str, float]):
            Dictionary containing RMSE and R Score
    """
    if np.isnan(y_true).any():
        raise ValueError("Input Real arrays contain NaN values.")
    elif np.isnan(y_pred).any():
        raise ValueError("Input Pred arrays contain NaN values.")
        
    
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r_socre = np.sqrt(r2) if r2 >= 0 else 0
    return rmse, r_socre


#* SEED GENERATION
def set_seed(seed: int) -> None:
    """
    Sets random seed for repoducibility.
    Ars:
        seed (int):
            Seed values for random number generation.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


#* EARLY STOPPING CLASS
class EarlyStopping:
    """
    Early Stopping to stop training when validation loss does not improve.
    Args:
        patience (int): 
            Number of epochs with no improvement after which training will be stopped.
        min_delta (float): 
            Minimum change in the monitored quantity to qualify as an improvement.
        verbose (bool): 
            If True, prints a message for each validation loss improvement.
    """
    def __init__(self, patience: int, min_delta: float = 0.0, verbose: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop_triggered = False
    
    def __call__(self, current_val_loss: float) -> bool:
        if current_val_loss < self.best_loss - self.min_delta:
            self.best_loss = current_val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop_triggered = True
                if self.verbose:
                    print(f"EarlyStopping: Validation loss did not improve for {self.patience} epochs. Stopping training.")
        return self.early_stop_triggered


#* TRAINING AND VALIDATION FUNCTION
def train_val_model(model: nn.Module, criterion: nn.Module, train_loader: DataLoader, val_loader: Union[DataLoader, None], config: Dict[str, Any], paths: Dict[str, Path], delay: int, device: Union[str, torch.device], seed: int = 42, fold_identifier: str = "") -> Tuple[nn.Module, pd.DataFrame, float, Union[Path, None]]:
    """
    Train and validate the model. Can also be used for final training if val_loader is None.
    Args:
        model (nn.Module):
            Neural network model to train.
        criterion (nn.Module):
            Loss function to optimize.
        train_loader (DataLoader):
            DataLoader for training data.
        val_loader (Union[DataLoader, None]):
            DataLoader for validation data. If None, only training is performed.
        config (Dict[str, Any]):
            Configuration dictionary containing training parameters and model settings.
        paths (Dict[str, Path]):
            Dictionary of paths for saving model and results.
        delay (int):
            Number of time steps to look back in the input sequence.
        device (Union[str, torch.device]):
            Device to run the training on ('cuda', 'cpu', 'mps').
        seed (int, optional):
            Random seed for reproducibility. Defaults to 42.
        fold_identifier (str, optional):
            Identifier for the current fold in cross-validation.    
    
    Returns:
        Tuple ([nn.Module, pd.DataFrame, float, Path]):
            Trained model with the best weights found during training, DataFrame containing training and validation metrics history for all epochs, best validation RMSE achieved, and the path where the model was saved.
    """
    set_seed(seed)
    EPOCH = config['constant']['EPOCH']

    # Hyperparameters
    lr = config['nn']['lr']
    optimizer_type = config['nn']['optimizer_type']
    schler = config['nn']['schler']
    schler_patience = config['nn']['patience_schler']
    early_patience = config['nn']['patience']

    # Information
    type_model = config['nn']['type_model']
    auroral_index = config['data']['auroral_index']
    run_specific_tag = f"_fold_{fold_identifier}" if fold_identifier else "_final"
    model_save_filname = f"{type_model}_{auroral_index}_delay_{delay}{run_specific_tag}.pt"
    model_save_path = paths['models_file'] / model_save_filname
    result_path = paths['metrics_result_file'] / f"metrics_train_val_{type_model}_{auroral_index}_{delay}_{fold_identifier}.csv"

    # Optimizer selection
    if optimizer_type.upper() == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-4)
    elif optimizer_type.upper() == 'ADAMW':
        optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Choose 'Adam' or 'AdamW' ")
    
    #Scheduler selection
    match schler.strip().lower():
        case 'reduce': 
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.3, patience = schler_patience)
        case 'cycle':
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, steps_per_epoch = len(train_loader), epochs = 50)
        case 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCH, eta_min = 0)
        case 'cosinerw':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = EPOCH, T_mult = 1)
        case _:
            print(f"Warning: Unsupported scheduler name '{schler.upper()}'. No scheduler will be used.")


    # Early stopping
    early_stopper = None
    if val_loader:
        early_stopper = EarlyStopping(patience = early_patience, verbose = True)
    
    best_model_weights = deepcopy(model.state_dict())
    best_val_loss = float('inf') if val_loader else 0

    metrics_log = {'train_rmse': [], 'train_r_score': []}
    if val_loader:
        metrics_log.update({'val_rmse': [], 'val_r_score': []})

    print(f"\n{'=' * 5} Training Model: {type_model} for {auroral_index} with Delay {delay} {fold_identifier} {'=' * 5}")

    # Training and Validation loop
    for epoch in range(EPOCH):
        model.train()
        train_real, train_pred = [], []
        train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_real.extend(y.detach().squeeze(-1).cpu().numpy())
            train_pred.extend(yhat.detach().squeeze(-1).cpu().numpy())

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_metrics = calculate_metrics(np.array(train_real), np.array(train_pred))
        metrics_log['train_rmse'].append(train_metrics[0])
        metrics_log['train_r_score'].append(train_metrics[1])


        if val_loader:
            model.eval()
            val_real, val_pred = [], []
            val_loss = 0.0
            
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    yhat = model(x)
                    loss = criterion(yhat, y)
                    
                    val_loss += loss.item() * x.size(0)
                    val_real.extend(y.detach().squeeze(-1).cpu().numpy())
                    val_pred.extend(yhat.detach().squeeze(-1).cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_rmse, val_r = calculate_metrics(np.array(val_real), np.array(val_pred))
            metrics_log['val_rmse'].append(val_rmse)
            metrics_log['val_r_score'].append(val_r)


        if (epoch + 1) % 10 == 0 or epoch == epoch - 1: 
            print(f"\n--- Epoch {epoch:03d}/{EPOCH} ---")
            print(f"Train | RMSE: {train_metrics[0]:.4f} | R Score: {train_metrics[1]:.4f}")
            if val_loader:
                print(f"Valid | RMSE: {val_rmse:.4f} | R Score: {val_r:.4f}")
        
        if scheduler and val_loader:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss) 
            else: scheduler.step() 
        elif scheduler and not val_loader: 
             if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): scheduler.step()

        if val_loader and early_stopper: 
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_weights = deepcopy(model.state_dict())
                if (epoch + 1) % 2 == 0 or epoch == EPOCH - 1:
                    print(f"Epoch {epoch+1}: Val loss improved to {avg_val_loss:.4f}. Model snapshot updated.")
            if early_stopper(avg_val_loss): 
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        elif not val_loader: 
            best_model_weights = deepcopy(model.state_dict())


    print(f"\nTraining finished.")
    if val_loader: 
        print(f"Best validation loss achieved: {best_val_loss:.4f}")
              
    model.load_state_dict(best_model_weights) 
    torch.save(best_model_weights, model_save_path)
    print(f"Model saved to {model_save_path}")

    num_logged_epochs = len(metrics_log['train_rmse'])
    metrics_history_df_data = {}
    for key, value_list in metrics_log.items():
        metrics_history_df_data[f'{key}_{delay}{"_" + fold_identifier if fold_identifier else ""}'] = value_list[:num_logged_epochs]
    metrics_history_df = pd.DataFrame(metrics_history_df_data)

    metrics_history_df.to_csv(result_path, index = False)

    return model, metrics_history_df, best_val_loss, model_save_path


#* TESTING FUNCTION
def testing_model(model: nn.Module, criterion: Union[nn.Module, None], test_loader: DataLoader, config: Dict[str, Any], paths: Dict[str, Path], best_model_file: Dict[str, Path], delay: int, test_epoch: pd.Series, device: Union[str, torch.device], is_final_test: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a trained model on a test dataset and return predictions and metrics.

    Loads the model weights from disk, performs inference on the test set, and computes evaluation metrics (RMSE and R Score). Handles both standard test and final evaluation scenarios. Returns both the predictions and the test metrics as DataFrames.

    Args:
        model (nn.Module):
            Instantiated model architecture (weights will be loaded from file).
        criterion (Union[nn.Module, None]):
            Loss function used for evaluation (can be None if not needed).
        test_loader (DataLoader):
            DataLoader for the test dataset.
        config (Dict[str, Any]):
            Configuration dictionary containing model and data settings.
        paths (Dict[str, Path]):
            Dictionary of paths for loading model and saving results.
        delay (int):
            Number of time steps to look back in the input sequence.
        test_epoch (pd.Series):
            Series of epoch or time indices corresponding to the test samples.
        device (Union[str, torch.device]):
            Device to run the evaluation on ('cuda', 'cpu', 'mps').
        is_final_test (bool, optional):
            If True, loads the final evaluation model. Defaults to False.

    Returns:
        Tuple ([pd.DataFrame, pd.DataFrame]):
            DataFrame with columns ['Epoch', 'auroral_index_real', 'auroral_index_pred'] containing true and predicted values for the test set and DataFrame with test metrics (RMSE and R Score) for the evaluated model.
    """
    type_model = config['nn']['type_model']
    auroral_index = config['data']['auroral_index']
    model_tag = "final" if is_final_test else "test"

    model_load_filename = best_model_file
    model_load_path = paths['models_file'] / model_load_filename
    
    pred_filname = paths['result_file'] / f"predictions_test_delay{delay}.csv"

    if not model_load_path.exists():
        # Fallback for models saved without a specific tag (e.g. if not using the CV workflow strictly)
        model_load_filename_fallback = f"{type_model}_{auroral_index}_delay_{delay}.pt"
        model_load_path_fallback = paths['models_file'] / model_load_filename_fallback
        if model_load_path_fallback.exists():
            print(f"Primary model file {model_load_path} not found. Using fallback: {model_load_path_fallback}")
            model_load_path = model_load_path_fallback
        else:
            raise FileNotFoundError(f"Model file not found for testing: {model_load_path} or {model_load_path_fallback}")
        
    print(f"\n===== Testing Model: {type_model} for {auroral_index} with Delay {delay} ({'Final Evaluation' if is_final_test else 'Test Run'}) =====")
    print(f"Loading model from: {model_load_path}\n")

    try:
        # It's better if main.py instantiates the model shell first using type_nn, then passes it here.
        # This function would then just load the state_dict.
        # If `model` is already the trained one, this just confirms.
        model.load_state_dict(torch.load(model_load_path, map_location=device))
    except Exception as e: # Catch a broader range of errors during load_state_dict
        print(f"Error loading state_dict into the provided model instance: {e}")
        print("Ensure the `model` argument to `model_testing` is the correct, instantiated architecture shell if not already trained.")
        raise e
    
    model.to(device) 
    model.eval() 
    test_real, test_pred = [], []
    test_loss = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss = criterion(yhat, y)
            test_loss += loss.item() * x.size(0)

            test_pred.append(yhat.detach().squeeze(-1).cpu().numpy())
            test_real.append(y.detach().cpu().numpy())

    
    test_metrics = calculate_metrics(np.concatenate(test_real, axis = 0), np.concatenate(test_pred, axis = 0))
    avg_test_loss_str = f"{test_loss / len(test_loader.dataset):.4f}" if criterion and len(test_loader.dataset) > 0 else "N/A"

    print(f"\nTest Results ---")
    if criterion: 
        print(f"Average Test Loss: {avg_test_loss_str}")
    print(f"Test RMSE: {test_metrics[0]:.4f} | R_Score: {test_metrics[1]:.4f}\n")

    metric_col_suffix = f"_{delay}" 
    if is_final_test: 
        metric_col_suffix += "_final_eval" # More descriptive for final eval metrics

    test_metrics_df = pd.DataFrame({
        f'Test_RMSE{metric_col_suffix}': [test_metrics[0]],
        f'Test_R_Score{metric_col_suffix}': [test_metrics[1]]
    })
    np_real = np.concatenate(test_real, axis = 0).tolist()
    np_pred = np.concatenate(test_pred, axis = 0).tolist()

    result_df = pd.DataFrame({
        'Epoch': test_epoch,
        f'{auroral_index}_real': np.concatenate(test_real, axis = 0).tolist(),
        f'{auroral_index}_pred': np.concatenate(test_pred, axis = 0).tolist()
    })

    if auroral_index == 'AL_INDEX':
        result_df[f'{auroral_index}_real'] = - 1 * result_df[f'{auroral_index}_real']
        result_df[f'{auroral_index}_pred'] = - 1 * result_df[f'{auroral_index}_pred']
   
    result_df.to_csv(pred_filname, index = False)

    return result_df, test_metrics_df

