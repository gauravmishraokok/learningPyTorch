from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
import warnings
from pathlib import Path
import torch
from modules._12_bilingualDataset import BilingualDataset
from modules._13_buildTokenizer_DataLoader_and_Transformer import get_or_build_tokenizer, get_model,get_ds
from modules._14_config import get_config , get_weights_file_path

def train_model(config):
    
    #Defining the device -> 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, test_dataloader, tokenizer_src , tokenizer_tgt = get_ds(config=config)
    
    model = get_model(config=config, vocab_src_len= tokenizer_src.get_vocab_size() , vocab_tgt_len= tokenizer_tgt.get_vocab_size()).to(device)
    
    #TensorBoard -> 
    writer = SummaryWriter(config['experiment_name'])
    #Optimizer -> 
    optimizer = torch.optim.Adam(params=model.parameters(),lr=config['lr'],eps=1e-9)
    
    intial_step = 0
    global_step = 0
    best_accuracy = 0.0
    
    if config['preload']:
        model_filename = get_weights_file_path(config,config['preload'])
        print(f"Preloading model name -> {model_filename}")
        state = torch.load(model_filename)
        intial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing=0.1).to(device)
    
    #Training Loop -> 
    
    for epoch in range(intial_epoch,config['num_epochs']):
        
        model.train()
        batch_iterator = tqdm(train_dataloader,desc=f'Processing epoch {epoch:02d}')
        
        
        for batch in batch_iterator:
            
            encoder_input = batch["encoder_input"].to(device) #(Batch , Seq_len)     
            decoder_input = batch["decoder_input"].to(device) #(Batch , Seq_len)
            
            encoder_mask = batch['encoder_mask'].to(device)  #(Batch , 1 ,  1 , Seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  #(Batch , 1 ,  Seq_len , Seq_len)
            
            #Passing the tensors through the transformer -> 
            encoder_output = model.encode(src = encoder_input,src_mask = encoder_mask) #(Batch , Seq_len , d_model)
            decoder_output = model.decode(tgt = decoder_input,encoder_output = encoder_output,src_mask = encoder_mask,tgt_mask = decoder_mask) #(Batch , Seq_len , d_model)
            project_output = model.projection_layer(decoder_output) #(Batch , Seq_len tgt_vocab_size)  
            
            
            label = batch['label'].to(device) #(Batch , Seq_len)
            
            #
            loss = loss_fn(project_output.view(-1,tokenizer_tgt.get_vocab_size()))  #(Batch , Seq_len , tgt_vocab_size) -> (Batch * Seq_len, tgt_vocab_size)         
            
            batch_iterator.set_postfix({f"Loss" : f"{loss.item():6.3f}"})
            
            
            #Logging the loss -> 
            writer.add_scalar('Train Loss', loss.item(),global_step)
            writer.flush()
            
            #Backpropagation -> 
            loss.backward()
            
            #Updating the weights -> 
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1 
            
            
        
        #Saving the model ->
        model_filename = get_weights_file_path(config,f"{epoch:02d}")
        
        torch.save(
            {
                "epoch" : epoch,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "global_step" : global_step,
                
            },model_filename)
        
    
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
