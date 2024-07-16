import torch
from torch.utils.data import DataLoader
from custom_datasets.flickr30k import Flickr30kDataset
from checkpoint_handler import load_model_checkpoint
from transformers import BlipForConditionalGeneration, AutoProcessor
from logger import CaptionLogger

def generate_captions(checkpoint_dir, root_dir, annotations_file, log_file, batch_size=1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
    
    model, processor, _, _, _, _ = load_model_checkpoint(checkpoint_dir)
    #processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    #model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    
    model.eval()
    
    dataset = Flickr30kDataset(root_dir=root_dir, annotations_file=annotations_file, processor=processor)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    
    logger = CaptionLogger(log_file)
    processed_images = set()
    captions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            img_names = batch["img_name"]
            pixel_values = batch["pixel_values"].to(device)

            unique_img_names = [img_name for img_name in img_names if img_name not in processed_images]
            if not unique_img_names:
                continue  # Skip if all images in the batch have been processed

            unique_pixel_values = [pixel_values[i] for i, img_name in enumerate(img_names) if img_name not in processed_images]

            if not unique_pixel_values:
                continue  # Skip if no unique images in the batch

            # Convert list back to tensor
            unique_pixel_values = torch.stack(unique_pixel_values)

            # Generate captions
            #outputs = model.generate(pixel_values=unique_pixel_values, max_length=200, num_beams=10, early_stopping=True)
            outputs = model.generate(pixel_values=unique_pixel_values, num_beams=10, length_penalty=2, repetition_penalty=2.0, min_length=200, max_length=300)

            decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)

            for img_name, caption in zip(unique_img_names, decoded_outputs):
                log_message = f"Image Name: {img_name}, Caption: {caption}"
                print(log_message)
                logger.log(log_message)
                captions.append((img_name, caption))
                processed_images.add(img_name)
    
    return captions

if __name__ == "__main__":
    checkpoint_path = "./checkpoint_dir/checkpoint-step-1800"
    root_dir = "./flickr30k/flickr30k_images"
    annotations_file = "./flickr30k/results.csv"
    log_file = "./captions.log"
    
    captions = generate_captions(checkpoint_path, root_dir, annotations_file, log_file)
