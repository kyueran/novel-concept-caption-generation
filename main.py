import os
import torch
import numpy as np
import random
from transformers import AutoProcessor, BlipForConditionalGeneration
from torch.utils.data import DataLoader
from custom_datasets.flickr30k import Flickr30kDataset
from parsers import get_parser
from checkpoint_handler import save_model_checkpoint, load_model_checkpoint
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Create output and checkpoint directories if they don't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.reload_checkpoint and args.checkpoint_dir and not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    if args.reload_checkpoint:
        model, processor, optimizer_state, scheduler_state, start_epoch, global_step = load_model_checkpoint(args.checkpoint_dir)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(optimizer_state)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # Example scheduler
        scheduler.load_state_dict(scheduler_state)
    else:
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # Example scheduler
        start_epoch = 0
        global_step = 0

    if torch.cuda.device_count() > 1:
        print("Device Count:", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    model.to(device)

    full_dataset = Flickr30kDataset(root_dir=args.root_dir, annotations_file=args.annotations_file, processor=processor)
    # Split the dataset into training, validation, and test sets
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)
    
    writer = SummaryWriter(log_dir=args.log_dir)

    model.train()

    for epoch in range(start_epoch, args.epochs):
        print("Epoch:", epoch)
        total_train_loss = 0
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            attention_mask = batch.pop("attention_mask").to(device)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            labels=input_ids)
            
            loss = outputs.loss.mean()
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            if global_step % args.logging_steps == 0:
                writer.add_scalar("Loss/Train", loss.item(), global_step)

            print(f"Step [{idx+1}/{len(train_dataloader)}], Loss: {loss.item()}")

            # Save the model checkpoint every 100 steps
            if global_step % args.save_steps == 0:
                checkpoint_step_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                save_model_checkpoint(model, processor, optimizer, scheduler, checkpoint_step_dir, epoch, global_step)
                print(f"Model checkpoint saved at {checkpoint_step_dir}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                input_ids = batch.pop("input_ids").to(device)
                pixel_values = batch.pop("pixel_values").to(device)
                attention_mask = batch.pop("attention_mask").to(device)

                outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            labels=input_ids)
                
                loss = outputs.loss.mean()
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)

        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        model.train()
        
        # Save the model checkpoint
        if (epoch + 1) % args.save_epochs == 0:
            checkpoint_epoch_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            save_model_checkpoint(model, processor, optimizer, scheduler, checkpoint_epoch_dir, epoch, global_step)
            print(f"Model checkpoint saved at {checkpoint_epoch_dir}")

    writer.close()

if __name__ == "__main__":
    main()
