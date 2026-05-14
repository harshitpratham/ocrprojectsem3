import pandas as pd
import torch
from torch.utils.data import Dataset

class PreprocessedDataset(Dataset):  
    def __init__(self, meta_csv, processor, max_target_length=128):  
        self.df = pd.read_csv(meta_csv)  
        self.processor = processor  
        self.max_target_length = max_target_length  
  
    def __len__(self):  
        return len(self.df)  
  
    def __getitem__(self, idx):  
        tensor_path = self.df['tensor_path'][idx]  
        pixel_values = torch.load(tensor_path)  # [3, H, W]  
        text = self.df['text'][idx]  
        labels = self.processor.tokenizer(  
            text,  
            padding="max_length",  
            max_length=self.max_target_length  
        ).input_ids  
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]  
        return {  
            "pixel_values": pixel_values,  
            "labels": torch.tensor(labels)  
        }  
