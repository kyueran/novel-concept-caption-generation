import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Training parameters for caption generation")

    # Dataset parameters
    parser.add_argument("--root_dir", type=str, default="./flickr30k/images", help="Root directory of the images")
    parser.add_argument("--annotations_file", type=str, default="./flickr30k/captions.csv", help="Path to the annotations file")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every n steps")
    parser.add_argument("--save_epochs", type=int, default=1, help="Save checkpoint every n epochs")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Total number of checkpoints to keep")

    # Logging and evaluation
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every n steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluate every n steps")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for checkpoints and logs")
    parser.add_argument("--fp16", action='store_true', help="Use mixed precision training")
    parser.add_argument("--log_dir", type=str, default="./runs", help="Directory for TensorBoard logs")

    return parser
