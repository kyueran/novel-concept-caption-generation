import os
import json
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import spacy
from sklearn.cluster import KMeans
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Setup logging to log to both console and file
log_file = 'caption_generation.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
logger = logging.getLogger()

# Parameters
data_folder = 'final_dataset'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
checkpoint_file = 'BEST_34checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'

word_map_file = 'WORDMAP.json'
device = torch.device("cpu")
cudnn.benchmark = True

# Load model
torch.nn.Module.dump_patches = True
checkpoint = torch.load(checkpoint_file, map_location=device)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()

# Load word map (word2ix)
word_map_file = os.path.join(data_folder, word_map_file)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

nlp = spacy.load("en_core_web_sm")

def generate_caption(image_features, beam_size=5):
    """
    Generate caption for a single image

    :param image_features: pre-extracted image features
    :param beam_size: beam size at which to generate captions for evaluation
    :return: generated caption
    """
    k = beam_size

    # Move to device
    image_features = image_features.to(device)  # (1, 3, 256, 256)
    image_features_mean = image_features.mean(1).to(device)
    image_features_mean = image_features_mean.expand(k, 2048).to(device)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h1, c1 = decoder.init_hidden_state(k)  # (batch_size, decoder_dim)
    h2, c2 = decoder.init_hidden_state(k)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1).to(device)  # (s, embed_dim)
        h1, c1 = decoder.top_down_attention(
            torch.cat([h2.to(device), image_features_mean, embeddings], dim=1).to(device),
            (h1.to(device), c1.to(device)))  # (batch_size_t, decoder_dim)
        attention_weighted_encoding = decoder.attention(image_features, h1).to(device)
        h2, c2 = decoder.language_model(
            torch.cat([attention_weighted_encoding, h1], dim=1).to(device), (h2.to(device), c2.to(device)))

        scores = decoder.fc(h2).to(device)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1).to(device)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1).to(device)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h1 = h1[prev_word_inds[incomplete_inds]]
        c1 = c1[prev_word_inds[incomplete_inds]]
        h2 = h2[prev_word_inds[incomplete_inds]]
        c2 = c2[prev_word_inds[incomplete_inds]]
        image_features_mean = image_features_mean[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    # Hypothesis
    hypothesis = [rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
    hypothesis = ' '.join(hypothesis)
    return hypothesis

def process_image(image_features, image_name):
    results = []
    # Ensure the decoder runs on CPU
    decoder.to("cpu")
    image_features = torch.FloatTensor(image_features).unsqueeze(0).to("cpu")
    caption = generate_caption(image_features)
    results.append({"image_name": image_name, "comment_number": 0, "comment": caption + "."})
    return results

def get_processed_images(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep='|')
        if not df.empty:
            return set(df['image_name'])
    return set()

def generate_captions_for_images(npy_file_path, output_csv_path, batch_size=100):
    data = np.load(npy_file_path, allow_pickle=True)
    image_features_list = data['features']
    image_names = data['names']

    processed_images = get_processed_images(output_csv_path)
    start_index = 0
    while start_index < len(image_names) and image_names[start_index] in processed_images:
        start_index += 1

    logger.info(f"Starting from index: {start_index}")

    if not os.path.exists(output_csv_path):
        pd.DataFrame(columns=["image_name", "comment_number", "comment"]).to_csv(output_csv_path, index=False, sep='|')

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for i in range(start_index, len(image_features_list), batch_size):
            batch_futures = [executor.submit(process_image, image_features_list[j], image_names[j]) for j in range(i, min(i + batch_size, len(image_features_list)))]
            for future in tqdm(as_completed(batch_futures), total=len(batch_futures)):
                try:
                    batch_results = future.result()
                    df_batch = pd.DataFrame(batch_results)
                    df_batch.to_csv(output_csv_path, mode='a', header=False, index=False, sep='|')
                    logger.info(f"Processed up to image: {batch_results[-1]['image_name']}")
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")

if __name__ == '__main__':
    # Set multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn')

    npy_file_path = '../shared_data/flickr30k_name_features.npy'
    output_csv_path = '../shared_data/output_captions_simple.csv'

    generate_captions_for_images(npy_file_path, output_csv_path)

