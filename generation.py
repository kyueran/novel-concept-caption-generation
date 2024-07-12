import torch
from torch.utils.data import DataLoader
from custom_datasets.flickr30k import Flickr30kDataset
from checkpoint_handler import load_model_checkpoint
from transformers import BlipForConditionalGeneration, AutoProcessor
from logger import CaptionLogger

def generate_captions(checkpoint_dir, root_dir, annotations_file, log_file, batch_size=1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, processor, _, _, _, _ = load_model_checkpoint(checkpoint_dir)
    model.to(device)
    
    # Remove DataParallel to simplify debugging
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    
    model.eval()
    
    dataset = Flickr30kDataset(root_dir=root_dir, annotations_file=annotations_file, processor=processor)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    
    logger = CaptionLogger(log_file)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            logger.log(batch['img_name'])

    captions = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx}")
            pixel_values = batch["pixel_values"].to(device)
            img_names = batch["img_name"]

            # Debugging statement to check for duplicates
            if len(set(img_names)) != len(img_names):
                print(f"Duplicate image names in batch {batch_idx}: {img_names}")

            outputs = model.generate(pixel_values=pixel_values)
            decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)
            captions.extend(decoded_outputs)
            
            for img_name, caption in zip(img_names, decoded_outputs):
                log_message = f"Image Name: {img_name}, Caption: {caption}"
                print(log_message)
                logger.log(log_message)
    
    return captions

if __name__ == "__main__":
    checkpoint_path = "./checkpoint_dir/checkpoint-step-1000"
    root_dir = "./flickr30k/flickr30k_images"
    annotations_file = "./flickr30k/results.csv"
    log_file = "./captions.log"
    
    captions = generate_captions(checkpoint_path, root_dir, annotations_file, log_file)
