from pathlib import Path

def get_config():
    """
    Returns the configuration dictionary for the model training and evaluation.

    This configuration includes parameters such as batch size, number of epochs, learning rate,
    sequence length, model dimensions, language codes, dataset configuration, model folder paths,
    and other relevant settings.

    Returns:
        dict: Configuration dictionary containing model and training parameters.
    """
    return {
        "batch_size": 8,
        "num_epochs": 1,
        "lr": 0.001,
        "seq_len": 500,
        "d_model": 256,
        "lang_src": "en",
        "lang_tgt": "hi",
        "dataset_config": "default",
        "model_folder": "weights",
        "model_basename": "transformerModel",
        "preload": "latest",
        "N": 2,
        "h": 4,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/transformerModel"
    }

def get_weights_file_path(config, epoch: str):
    """
    Constructs the file path for the model weights file for a specific epoch.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        epoch (str): The epoch number or identifier to include in the filename.

    Returns:
        str: The file path for the model weights file.
    """    
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pth"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    """
    Finds the latest weights file in the specified weights folder.

    This function searches for all files matching the model basename pattern,
    sorts them, and returns the path to the most recent weights file.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        str or None: The file path for the latest weights file, or None if no weights files are found.
    """
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
