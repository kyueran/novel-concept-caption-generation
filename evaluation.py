from transformers import BertTokenizer
from evaluate import load

# Initialize the tokenizer and metric once
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
metric = load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip().split() for pred in decoded_preds]
    decoded_labels = [[label.strip().split()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["bleu"]}
