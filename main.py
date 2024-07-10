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

def main():
    parser = get_parser()
    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    full_dataset = Flickr30kDataset(root_dir=args.root_dir, annotations_file=args.annotations_file, transform=image_processor, tokenizer=tokenizer)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, f"vision-encoder-decoder-{current_time}")
    output_dir = os.path.join(args.output_dir, f"vision-encoder-decoder-{current_time}")

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        output_dir=output_dir,  # Specify the separate directory for outputs and checkpoints
        overwrite_output_dir=True,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        logging_dir=log_dir,
        report_to="tensorboard",
        num_train_epochs=args.epochs,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=lambda data: {"pixel_values": torch.stack([f["pixel_values"] for f in data]).to(device),
                                    "input_ids": torch.stack([f["input_ids"] for f in data]).to(device),
                                    "attention_mask": torch.stack([f["attention_mask"] for f in data]).to(device)},
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(TensorBoardCallback())

    trainer.train()

    # Save the final model
    trainer.save_model()

    # Evaluate the model on the test set
    test_results = trainer.evaluate(test_dataset)
    print(f"Test results: {test_results}")

    # To run TensorBoard, use the following command in your terminal:
    # tensorboard --logdir=runs

if __name__ == "__main__":
    main()
