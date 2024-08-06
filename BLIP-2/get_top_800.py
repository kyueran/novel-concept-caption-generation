import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import json
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval

def calculate_loss_per_image(model, data_loader, device):
    model.eval()
    image_caption_losses = {}

    with torch.no_grad():
        for images, captions, image_ids in tqdm(data_loader, desc="Processing batches"):
            images = images.to(device)
            
            for i in range(len(captions)):
                image = images[i].unsqueeze(0)  # Add batch dimension
                caption = captions[i]
                image_id = image_ids[i]
                loss = model(image, caption)

                if image_id in image_caption_losses:
                    image_caption_losses[image_id].append((loss.item(), caption))
                else:
                    image_caption_losses[image_id] = [(loss.item(), caption)]

    # Calculate mean loss per image ID
    mean_losses = [(np.mean([loss for loss, _ in losses]), image_id, [caption for _, caption in losses]) for image_id, losses in image_caption_losses.items()]
    
    # Sort by mean loss in descending order and select the top 800
    mean_losses.sort(reverse=True, key=lambda x: x[0])
    top_800_images = mean_losses[:800]

    return top_800_images

def save_top_800_images(images, output_dir, data_type):
    output_path = os.path.join(output_dir, f'flickr30k_{data_type}_top800.json')
    formatted_images = []
    for _, img, caps in images:
        for cap in caps:
            formatted_images.append({"image": img, "caption": cap})
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(formatted_images, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_image_ids(json_data):
    return {item['image'] for item in json_data}

def parse_captions(file_path):
    caption_data = {}
    with open(file_path, 'r') as file:
        next(file)  # Skip the header row
        for line in file:
            parts = line.strip().split('|')
            image_name = parts[0]
            comment = parts[2]
            if image_name not in caption_data:
                caption_data[image_name] = []
            caption_data[image_name].append(comment)
    return caption_data

class CustomImageDataset(Dataset):
    def __init__(self, image_folder, captions, transform=None, exclude_ids=None):
        self.image_folder = image_folder
        self.captions = captions
        self.transform = transform
        self.exclude_ids = exclude_ids

        self.data = [
            (img, cap, img) for img, caps in self.captions.items() if img not in self.exclude_ids for cap in caps[:2]
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption, img_id = self.data[idx]
        img_path = os.path.join(self.image_folder, img_name)
        
        # Debugging: print the image path
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, caption, img_id

def main(args, config):    
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    json_file_1 = '/home/kyueran/caption-generation/BLIP/annotation/f30k_human_rand100_val.json'
    json_file_2 = '/home/kyueran/caption-generation/BLIP/annotation/f30k_human_rand100_test.json'

    data_1 = load_json(json_file_1)
    data_2 = load_json(json_file_2)

    # Extract image IDs and put them in a set
    exclude_image_ids = extract_image_ids(data_1).union(extract_image_ids(data_2))
    print(f"Number of excluded image IDs: {len(exclude_image_ids)}")

    # Parse captions
    caption_file = '/home/kyueran/caption-generation/shared_data/output_captions.csv'
    captions = parse_captions(caption_file)

    # Create dataset and dataloader
    image_folder = '/home/kyueran/caption-generation/shared_data/flickr30k/flickr30k_images'
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])

    dataset = CustomImageDataset(image_folder=image_folder, captions=captions, transform=transform, exclude_ids=exclude_image_ids)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Size of DataLoader: {len(dataloader.dataset)}")

    # Print the first 3 lines of content in the DataLoader
    for i, (images, captions, img_ids) in enumerate(dataloader):
        if i >= 3:
            break
        print(f"Batch {i + 1}:")
        print(f"Images: {images.size()}")
        print(f"Captions: {captions}")
        print(f"Image IDs: {img_ids}")

    # Create model
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                         vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                         prompt=config['prompt'])

    model = model.to(device)

    # Calculate loss for each image and get top 800 images with highest loss
    top_800_images = calculate_loss_per_image(model, dataloader, device)
    save_top_800_images(top_800_images, args.output_dir, 'train')
    print("Top 800 images with highest loss saved to JSON file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_flickr30k.yaml')
    parser.add_argument('--output_dir', default='output/flickr30k_top800')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=9, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
