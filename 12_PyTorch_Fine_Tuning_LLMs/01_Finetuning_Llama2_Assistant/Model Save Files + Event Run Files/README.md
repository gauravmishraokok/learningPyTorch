# This folder would not actually exist, Instead there would 2 folders ->

## 1. Llama-2-7b-chat-finetuned

**This will have adapter configuration and model weights saved.**

## 2. Results

**This will again have 2 subfolders, One would be "Checkpoint-n" when n is the number of max_epochs, it will have all the save files including optimizer, tokenizer, model ' s save files. The other folder would be "runs" which will have TensorBoard event files for plotting.**

![File Structure](https://i.imgur.com/J5ZcDVj.png)

## Why this README.md instead of the files?

The file size is very high, Thus not pushing, This readme file is enough to understand folder structure for future references.
