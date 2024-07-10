import os
import torch
from torch.utils.data import DataLoader, Subset
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, BertTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers.integrations import TensorBoardCallback
from datetime import datetime
from custom_datasets.flickr30k import Flickr30kDataset
from parsers import get_parser
from evaluation import compute_metrics

def main():
    #parser = get_parser()
    #args = parser.parse_args()

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to use GPU 0

    # Initialize CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Move model to device
    #model.to(device)

if __name__ == "__main__":
    main()