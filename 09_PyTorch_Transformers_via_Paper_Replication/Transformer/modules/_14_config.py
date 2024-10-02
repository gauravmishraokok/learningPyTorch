from pathlib import Path

def get_config():
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
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pth"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
