import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from modules._10_transformer import Transformer
from modules._11_buildTransformer import build_transformer
from modules._12_bilingualDataset import BilingualDataset

def get_all_sentences(ds, lang):
    """
    Yields all sentences in the specified language from the dataset.

    Args:
        ds (Dataset): The dataset containing translation pairs.
        lang (str): The language code for the desired sentences.

    Yields:
        str: A sentence in the specified language.
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    Retrieves or builds a tokenizer for the specified language.

    If a tokenizer file exists, it loads the tokenizer from the file.
    Otherwise, it trains a new tokenizer on the dataset and saves it to a file.

    Args:
        config (dict): Configuration dictionary containing tokenizer file paths.
        ds (Dataset): The dataset containing translation pairs.
        lang (str): The language code for the tokenizer.

    Returns:
        Tokenizer: The tokenizer for the specified language.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer

def get_ds(config):
    """
    Loads the dataset, builds tokenizers, and splits the dataset into training and testing sets.

    Args:
        config (dict): Configuration dictionary containing dataset and tokenizer parameters.

    Returns:
        tuple: A tuple containing the training DataLoader, testing DataLoader, source tokenizer, and target tokenizer.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    ds_raw = load_dataset('cfilt/iitb-english-hindi', 'default', split="train[:1%]")
    
    # Building the tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])
    
    # Splitting the training and testing data into 80% & 20% split
    train_ds_size = int(0.8 * len(ds_raw))
    test_ds_size = len(ds_raw) - train_ds_size
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    train_ds_raw, test_ds_raw = random_split(ds_raw, [train_ds_size, test_ds_size])
    
    train_ds = BilingualDataset(ds=train_ds_raw, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt, src_lang=config["lang_src"], tgt_lang=config["lang_tgt"], seq_len=config["seq_len"])
    test_ds = BilingualDataset(ds=test_ds_raw, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt, src_lang=config["lang_src"], tgt_lang=config["lang_tgt"], seq_len=config["seq_len"])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=True)
    
    return train_dataloader, test_dataloader , tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Builds and returns the transformer model.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        vocab_src_len (int): The size of the source vocabulary.
        vocab_tgt_len (int): The size of the target vocabulary.

    Returns:
        nn.Module: The transformer model.
    """
    model = build_transformer(src_vocab_size=vocab_src_len, tgt_vocab_size=vocab_tgt_len, src_seq_len=config["seq_len"], tgt_seq_len=config["seq_len"], d_model=config["d_model"])
    
    return model
