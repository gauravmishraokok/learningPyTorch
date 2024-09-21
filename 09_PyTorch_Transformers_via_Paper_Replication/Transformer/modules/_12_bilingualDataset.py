import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 

class BilingualDataset(Dataset):
    """
    A custom Dataset class for bilingual translation tasks.

    Attributes:
        ds (list): The dataset containing source and target language pairs.
        tokenizer_src (Tokenizer): Tokenizer for the source language.
        tokenizer_tgt (Tokenizer): Tokenizer for the target language.
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.
        seq_len (int): The sequence length for padding/truncating.
        sos_token (torch.Tensor): Start of sequence token for the target language.
        eos_token (torch.Tensor): End of sequence token for the target language.
        pad_token (torch.Tensor): Padding token for the target language.
    """
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang , seq_len ):
        """
        Initializes the BilingualDataset.

        Args:
            ds (list): The dataset containing source and target language pairs.
            tokenizer_src (Tokenizer): Tokenizer for the source language.
            tokenizer_tgt (Tokenizer): Tokenizer for the target language.
            src_lang (str): Source language code.
            tgt_lang (str): Target language code.
            seq_len (int): The sequence length for padding/truncating.
        """
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.long)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.long)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.long)
        
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.ds)
    
    def causal_mask(self, size):
        """
        Creates a causal mask for the decoder.

        Args:
            size (int): The size of the mask.

        Returns:
            torch.Tensor: A causal mask of shape (size, size).
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.int)
        return mask == 0
    
    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the encoder input, decoder input, masks, label, source text, and target text.
        """
        # Getting the source and target sentences together and then splitting them
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_pad_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_pad_tokens = self.seq_len - len(dec_input_tokens) - 1
        
        if enc_num_pad_tokens < 0 or dec_num_pad_tokens < 0:
            raise ValueError("The input sentence is too long!")
        
        # Adding the SOS, EOS and Padding to encoder input
        encoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(enc_input_tokens, dtype=torch.long),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype=torch.long)
            ]
        ) 
        
        # Adding SOS token and padding to decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.long),
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.long)
            ]
        )
        
        # Adding EOS token and padding to label (What we expect as output from decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.long),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.long)
            ]
        )
        
        # Checking the sizes of the tensors
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).long(),  # (1,1,seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).long() & self.causal_mask(decoder_input.size(0)),# (1,1,seq_len)
            "label": label, #(seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
