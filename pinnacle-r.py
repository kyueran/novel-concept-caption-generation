import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, BertTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers.integrations import TensorBoardCallback
from datetime import datetime
from custom_datasets.flickr30k import Flickr30kDataset
from parsers import get_parser
from evaluation import compute_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AdjustTrainingCallback(TrainerCallback):
    def __init__(self, patience, batch_size, train_dataset, select_data_based_on_loss):
        self.patience = patience
        self.no_improvement = 0
        self.best_val_loss = float('inf')
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.select_data_based_on_loss = select_data_based_on_loss

    def on_evaluate(self, args, state, control, **kwargs):
        val_loss = kwargs['metrics']['eval_loss']
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.no_improvement = 0
        else:
            self.no_improvement += 1

        if self.no_improvement >= self.patience:
            print("Overfitting detected. Adjusting training and validation sets.")
            train_subset, val_dataset = self.select_data_based_on_loss(
                kwargs['model'], self.train_dataset, self.batch_size)
            
            kwargs['train_dataloader'].dataset = train_subset
            kwargs['eval_dataloader'].dataset = val_dataset
            control.should_training_stop = True
            self.no_improvement = 0

def compute_losses(model, data_loader, device):
    model.eval()
    losses = []
    for batch in data_loader:
        inputs = batch['pixel_values'].to(device)
        labels = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(pixel_values=inputs, labels=labels, decoder_attention_mask=attention_mask)
            loss = outputs.loss  # This should be a tensor with individual losses for each data point
        losses.extend(loss.cpu().numpy())  # Collect individual losses
    model.train()
    return losses

def main():
    parser = get_parser()
    args = parser.parse_args()

    set_seed(42)

    # Initialize CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Load the pre-trained models and image processor
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # Move model to device
    model.to(device)

    # Create the datasets
    full_dataset = Flickr30kDataset(root_dir=args.root_dir, annotations_file=args.annotations_file, transform=image_processor, tokenizer=tokenizer)

    # Split the dataset into training and test sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    def select_data_based_on_loss(model, dataset, batch_size, top_fraction=0.2, val_fraction=0.2):
        data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
        losses = compute_losses(model, data_loader, device)

        # Select top fraction of data with highest loss for training
        top_k = int(top_fraction * len(losses))
        top_k_indices = sorted(range(len(losses)), key=lambda i: losses[i], reverse=True)[:top_k]
        top_k_dataset = Subset(dataset, top_k_indices)

        # Select a validation set from the remaining data
        remaining_indices = [i for i in range(len(losses)) if i not in top_k_indices]
        val_k = int(val_fraction * len(remaining_indices))
        val_indices = remaining_indices[:val_k]
        val_dataset = Subset(dataset, val_indices)

        return top_k_dataset, val_dataset

    # Initial selection of data for training and validation
    train_subset, val_dataset = select_data_based_on_loss(model, train_dataset, args.batch_size)

    # Ensure the training subset is larger than the validation set
    train_subset_size = len(train_subset)
    val_dataset_size = len(val_dataset)
    print(f"Training subset size: {train_subset_size}")
    print(f"Validation set size: {val_dataset_size}")

    # Load data into DataLoader
    train_dataloader = DataLoader(train_subset, shuffle=True, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    # Define the training arguments
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, f"vision-encoder-decoder-{current_time}")

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        logging_dir=log_dir,
        report_to="tensorboard",
        num_train_epochs=args.epochs,
    )

    # Define the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_dataset,
        data_collator=lambda data: {"pixel_values": torch.stack([f["pixel_values"] for f in data]).to(device),
                                    "input_ids": torch.stack([f["input_ids"] for f in data]).to(device),
                                    "attention_mask": torch.stack([f["attention_mask"] for f in data]).to(device)},
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(TensorBoardCallback())
    trainer.add_callback(AdjustTrainingCallback(patience=3, batch_size=args.batch_size, train_dataset=train_dataset, select_data_based_on_loss=select_data_based_on_loss))

    trainer.train()

    # Save the final model
    trainer.save_model()

    # Save the final model
    trainer.save_model()

    # Evaluate the model on the test set
    test_results = trainer.evaluate(test_dataset)
    print(f"Test results: {test_results}")

    # To run TensorBoard, use the following command in your terminal:
    # tensorboard --logdir=runs

if __name__ == "__main__":
    main()
