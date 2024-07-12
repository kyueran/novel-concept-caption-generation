import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class Flickr30kDataset(Dataset):
    def __init__(self, root_dir, annotations_file, processor=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotations_file, sep='|')
        self.processor = processor

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0].strip()
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        text = self.annotations.iloc[idx, 2].strip()

        encoding = self.processor(images=image, text=text, padding="max_length", return_tensors="pt")
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding['img_name'] = img_name
        return encoding