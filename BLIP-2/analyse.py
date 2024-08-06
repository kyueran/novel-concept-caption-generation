'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
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
from torch.utils.tensorboard import SummaryWriter

from models.blip_distil import blip_decoder
import utils
from utils import cosine_lr_schedule
from data import create_distillation_dataset, create_sampler, create_loader
from data.utils import save_result, flickr30k_caption_eval
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

def train(model, data_loader, optimizer, epoch, device, writer):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)       
        
        loss = model(image, caption)      
        writer.add_scalar('Train/Loss', loss.item(), epoch * len(data_loader) + i)
        writer.add_scalar('Train/Learning Rate', optimizer.param_groups[0]["lr"], epoch * len(data_loader) + i)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

def load_demo_image(image_size,device):
    img_url = '/home/kyueran/caption-generation/BLIP/merlion.jpg' 
    raw_image = Image.open(img_url).convert('RGB')   
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

@torch.no_grad()
def evaluate(model, data_loader, device, config, result_dir):
    # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    images_dir = os.path.join(result_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        
        image = image.to(device)       
        
        for i in range(len(image)):
            img = transforms.ToPILImage()(image[i].cpu())
            image_save_path = os.path.join(images_dir, f"{image_id[i]}.png")
            img.save(image_save_path)

        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                  min_length=config['min_length'])
        
        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})
  
    return result

def save_result_simple(result, output_dir):
    """
    Save the result of the evaluation to a JSON file.
    
    Args:
        result (list): List of dictionaries containing image_id and caption.
        output_dir (str): Directory where the result file will be saved.
        filename (str): Name of the result file (without extension).
        
    Returns:
        str: Path to the saved result file.
    """
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"results.json")
    
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=4)
    
    return result_file

def main(args, config):    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    writer = SummaryWriter(log_dir=args.output_dir)

    #### Dataset #### 
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_distillation_dataset('caption_flickr30k', config)  

    samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[4,4,4],
                                                          is_trains=[True, False, False], collate_fns=[None,None,None])         

    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])

    model = model.to(device)   
    
    model_without_ddp = model
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
            
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):      
        test_result = evaluate(model_without_ddp, test_loader, device, config, args.result_dir)  
        test_result_file = save_result_simple(test_result, args.result_dir)  

        coco_test = flickr30k_caption_eval(config['coco_gt_root'],test_result_file,'test')
        
        if args.evaluate:            
            log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()},
                         'score': coco_test.eval['CIDEr'] + coco_test.eval['Bleu_4'],
                        }
            with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                   
        else:             
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }

            log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()},                       
                            'epoch': epoch,
                            'best_score': best,
                            'best_epoch': best_epoch,
                        }
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")     
                    
        if args.evaluate: 
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/analyse_flickr30k.yaml')
    parser.add_argument('--output_dir', default='output/merlion_front_distil')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=9, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)