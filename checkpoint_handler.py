import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, BertTokenizer

def save_model_checkpoint(model, output_dir):
    """
    Save the model checkpoint.
    """
    model.save_pretrained(output_dir)
    print(f"Model checkpoint saved to {output_dir}")

def load_model_checkpoint(output_dir):
    """
    Load the model from a checkpoint.
    """
    model = VisionEncoderDecoderModel.from_pretrained(output_dir)
    feature_extractor = ViTFeatureExtractor.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print(f"Model loaded from {output_dir}")
    return model, feature_extractor, tokenizer

if __name__ == "__main__":
    # Example usage
    checkpoint_dir = "./results"

    # Save model checkpoint
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    save_model_checkpoint(model, checkpoint_dir)

    # Load model checkpoint
    loaded_model, loaded_feature_extractor, loaded_tokenizer = load_model_checkpoint(checkpoint_dir)
