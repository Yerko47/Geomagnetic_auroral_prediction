from pathlib import Path

#* STRUCTURE PROJECT
def structure_project(project_root = None):
    """
    Create project structure

    Args:
        project_root (str):
            Path to the project
    
    Returns:
        paths (dict):
            Dictionary with the project structure path
    """
    base = Path(project_root or Path(__file__).resolve().parent.parent)
    
    paths = {
        # Project File
        "raw_file": base / "data" / "raw",
        "processed_file": base / "data" / "processed",
        "config_file": base / "config",
        "models_file": base / "models",
        "result_file": base / "models" / "results_data",
        "metrics_result_file": base / "models" / "metrics_data",
        "test_file": base / "test",
        "docs_file": base / "docs",
        "notebook_file": base / "notebooks",

        # Project Plot file
        "historic_index_file": base / "plots" / "historic" / "index",
        "historic_solar_file": base / "plots" / "historic" / "solar",
        "stadistic_file": base / "plots" / "stadistics",
        "training_rmse": base / "plots" / "training" / "rmse",
        "training_rscore": base / "plots" / "training" / "rscore",
        "test_rmse": base / "plots" / "testing" / "metrics" / "rmse",
        "test_rscore": base / "plots" / "testing" / "metrics" / "rscore",
        "test_comparison": base / "plots" / "testing" / "comparison",
        "test_gift": base / "plots" / "testing" / "gift"
        
    }

    for p in paths.values():
        p.mkdir(parents = True, exist_ok = True)

    return paths

