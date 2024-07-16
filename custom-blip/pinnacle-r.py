import os
import torch
import numpy as np
import random
from transformers import AutoProcessor, BlipForConditionalGeneration
from torch.utils.data import DataLoader, Subset
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

def compute_losses(model, dataloader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            print(batch["img_name"])
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            labels=input_ids)
            
            loss = outputs.loss
            print("LOSS", loss)
            losses.extend(loss.cpu().numpy())
    model.train()
    return losses

def select_top_k(dataset, losses, top_fraction):
    top_k = int(top_fraction * len(losses))
    top_k_indices = sorted(range(len(losses)), key=lambda i: losses[i], reverse=True)[:top_k]
    top_k_dataset = Subset(dataset, top_k_indices)
    return top_k_dataset

def update_datasets(model, full_train_dataset, batch_size, device):
    initial_dataloader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=False)
    initial_losses = compute_losses(model, initial_dataloader, device)

    top_train_dataset = select_top_k(full_train_dataset, initial_losses, top_fraction=0.2)
    train_dataloader = DataLoader(top_train_dataset, batch_size=batch_size, shuffle=True)

    remaining_indices = [i for i in range(len(initial_losses)) if i not in top_train_dataset.indices]
    remaining_dataset = Subset(full_train_dataset, remaining_indices)
    val_subset_size = int(0.2 * len(remaining_dataset))
    val_subset_indices = random.sample(remaining_dataset.indices, val_subset_size)
    val_subset_dataset = Subset(remaining_dataset, val_subset_indices)
    val_dataloader = DataLoader(val_subset_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

def main():
    parser = get_parser()
    args = parser.parse_args()

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
        for state in optimizer.state.values():
            for k, v in state.items():
                #print("Optimizer: Key - ", k, "Value - ", v)
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        scheduler.load_state_dict(scheduler_state)
    else:
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        start_epoch = 0
        global_step = 0

    if torch.cuda.device_count() > 1:
        print("Device Count:", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    model.to(device)

    full_dataset = Flickr30kDataset(root_dir=args.root_dir, annotations_file=args.annotations_file, processor=processor)
    full_dataset_size = len(full_dataset)
    train_size = int(0.8 * full_dataset_size)
    indices = list(range(full_dataset_size))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    train_dataloader, val_dataloader = update_datasets(model, train_dataset, args.batch_size, device)

    writer = SummaryWriter(log_dir=args.log_dir)
    
    model.train()
    best_val_loss = float('inf')
    no_improvement = 0
    patience = 3

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

            print(f"Step [{global_step}/{len(train_dataloader)}], Loss: {loss.item()}")

            if global_step % args.save_steps == 0:
                checkpoint_step_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                save_model_checkpoint(model, processor, optimizer, scheduler, checkpoint_step_dir, epoch, global_step)
                print(f"Model checkpoint saved at {checkpoint_step_dir}")

            if global_step % 20 == 0:
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for idx, val_batch in enumerate(val_dataloader):
                        input_ids = val_batch.pop("input_ids").to(device)
                        pixel_values = val_batch.pop("pixel_values").to(device)
                        attention_mask = val_batch.pop("attention_mask").to(device)

                        outputs = model(input_ids=input_ids,
                                    pixel_values=pixel_values,
                                    attention_mask=attention_mask,
                                    labels=input_ids)
                        
                        loss = outputs.loss.mean()
                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)
                writer.add_scalar("Loss/Val_Step", avg_val_loss, global_step)

                print(f"Validation Loss: {avg_val_loss}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement >= patience:
                    print("Overfitting detected. Recomputing losses and updating training and validation sets.")
                    train_dataloader, val_dataloader = update_datasets(model, train_dataset, args.batch_size, device)
                    no_improvement = 0

                model.train()

        avg_train_loss = total_train_loss / len(train_dataloader)
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss}")

        checkpoint_epoch_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        save_model_checkpoint(model, processor, optimizer, scheduler, checkpoint_epoch_dir, epoch, global_step)
        print(f"Model checkpoint saved at {checkpoint_epoch_dir}")

    writer.close()

if __name__ == "__main__":
    main()
