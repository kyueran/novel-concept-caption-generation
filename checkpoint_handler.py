import os
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration

def save_model_checkpoint(model, processor, optimizer, scheduler, output_dir, epoch, global_step):
    if isinstance(model, torch.nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model
    
    # Ensure the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_to_save.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    checkpoint = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'global_step': global_step
    }
    torch.save(checkpoint, os.path.join(output_dir, 'training_state.pth'))



def load_model_checkpoint(checkpoint_dir):
    """
    Load the model, processor, optimizer, and scheduler from a checkpoint.
    """
    model = BlipForConditionalGeneration.from_pretrained(checkpoint_dir)
    processor = AutoProcessor.from_pretrained(checkpoint_dir)
    
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'training_state.pth'))
    optimizer_state = checkpoint['optimizer']
    scheduler_state = checkpoint['scheduler']
    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    
    print(f"Model, processor, optimizer, and scheduler loaded from {checkpoint_dir}")
    return model, processor, optimizer_state, scheduler_state, epoch, global_step