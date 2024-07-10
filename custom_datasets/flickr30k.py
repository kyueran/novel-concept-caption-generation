import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class Flickr30kDataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None, tokenizer=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotations_file, sep='|')
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0].strip())
        image = Image.open(img_name).convert("RGB")
        caption = self.annotations.iloc[idx, 2].strip()

        if self.transform:
            image = self.transform(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        if self.tokenizer:
            tokenized_caption = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            input_ids = tokenized_caption.input_ids.squeeze()
            attention_mask = tokenized_caption.attention_mask.squeeze()

        return {"pixel_values": image, "input_ids": input_ids, "attention_mask": attention_mask}
