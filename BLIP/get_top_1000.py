import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval
from tqdm import tqdm

def calculate_loss_per_caption(model, data_loader, device):
    model.eval()
    caption_losses = []

    with torch.no_grad():
        for image, caption, _ in tqdm(data_loader, desc="Processing batches"):
            image = image.to(device)
            loss = model(image, caption)
            caption_losses.append((loss.item(), image, caption))
    # Sort captions by loss in descending order and select the top 1000
    caption_losses.sort(reverse=True, key=lambda x: x[0])
    top_1000_captions = caption_losses[:1000]

    return top_1000_captions

def save_top_1000_captions(captions, output_dir, data_type):
    output_path = os.path.join(output_dir, f'flickr30k_{data_type}_top1000.json')
    formatted_captions = [{"image": image, "caption": caption} for _, image, caption in captions]
    
    with open(output_path, 'w') as f:
        json.dump(formatted_captions, f, indent=4)

def main(args, config):    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('caption_flickr30k', config)

    samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size']]*3, num_workers=[4,4,4],
                                                          is_trains=[True, False, False], collate_fns=[None, None, None])

    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                         vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                         prompt=config['prompt'])

    model = model.to(device)

    # Calculate loss for each caption and get top 1000 captions with highest loss
    top_1000_captions = calculate_loss_per_caption(model, train_loader, device)
    save_top_1000_captions(top_1000_captions, args.output_dir, 'train')
    print("Top 1000 captions with highest loss saved to JSON file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_flickr30k.yaml')
    parser.add_argument('--output_dir', default='output/flickr30k_top1000')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
