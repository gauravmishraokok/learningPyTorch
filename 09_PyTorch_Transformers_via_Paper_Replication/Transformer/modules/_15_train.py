import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from modules._13_buildTokenizer_DataLoader_and_Transformer import get_or_build_tokenizer, get_model, get_ds
from modules._14_config import get_config, get_weights_file_path

def latest_weights_file_path(config):
    """
    Find the latest weights file in the weights folder.

    Args:
        config (dict): Configuration dictionary containing model folder information.

    Returns:
        str or None: Path to the latest weights file or None if no files are found.
    """
    model_folder = Path(config['model_folder'])
    model_files = list(model_folder.glob('*.pth'))
    if not model_files:
        print(f"No model files found in {model_folder}")
        return None
    return max(model_files, key=lambda x: x.stat().st_mtime)

def train_model(config):
    """
    Train the Transformer model based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing training parameters.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config=config)
    
    model = get_model(config=config, vocab_src_len=tokenizer_src.get_vocab_size(), vocab_tgt_len=tokenizer_tgt.get_vocab_size()).to(device)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    if preload == 'latest':
        model_filename = latest_weights_file_path(config=config)
    elif preload:
        model_filename = get_weights_file_path(config, preload)
    else:
        model_filename = None

    if model_filename and Path(model_filename).exists():
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No valid model to preload, starting from scratch')
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    print(f"Starting training from epoch {initial_epoch}")
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        
        total_loss = 0
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            
            encoder_output = model.encode(src=encoder_input, src_mask=encoder_mask)
            decoder_output = model.decode(tgt=decoder_input, encoder_output=encoder_output, src_mask=encoder_mask, tgt_mask=decoder_mask)
            project_output = model.projection_layer(decoder_output)
            
            label = batch['label'].to(device)
            
            loss = loss_fn(project_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            total_loss += loss.item()
            batch_iterator.set_postfix({"Loss": f"{loss.item():6.3f}"})
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
        
        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {average_loss:.4f}")
        
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "loss": average_loss
        }, model_filename)
        print(f"Model saved: {model_filename}")

if __name__ == "__main__":
    config = get_config()
    train_model(config)
