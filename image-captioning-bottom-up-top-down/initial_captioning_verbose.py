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
log_file = '../shared_data/caption_generation.log'
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

def generate_caption(image_features, beam_size=10, length_penalty=1.2, diversity_penalty=1, n_clusters=2):
    k = beam_size

    # Cluster image features
    features = image_features.squeeze(0).cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    cluster_labels = kmeans.labels_

    def generate_partial_caption(cluster_features, k, length_penalty):
        cluster_features = torch.FloatTensor(cluster_features).unsqueeze(0).to(device)
        cluster_features_mean = cluster_features.mean(1).to(device)
        cluster_features_mean = cluster_features_mean.expand(k, 2048).to(device)

        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)
        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        h1, c1 = decoder.init_hidden_state(k)
        h2, c2 = decoder.init_hidden_state(k)

        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1).to(device)
            h1, c1 = decoder.top_down_attention(
                torch.cat([h2.to(device), cluster_features_mean, embeddings], dim=1).to(device),
                (h1.to(device), c1.to(device)))
            attention_weighted_encoding = decoder.attention(cluster_features, h1).to(device)
            h2, c2 = decoder.language_model(
                torch.cat([attention_weighted_encoding, h1], dim=1).to(device), (h2.to(device), c2.to(device)))

            scores = decoder.fc(h2).to(device)
            scores = F.log_softmax(scores, dim=1).to(device)
            scores = top_k_scores.expand_as(scores) + scores
            scores = scores / (step ** length_penalty)

            if step > 1:
                for i in range(scores.size(0)):
                    for j in range(scores.size(1)):
                        if j in seqs[i]:
                            scores[i, j] -= diversity_penalty

            # Penalize <unk> token
            unk_token_index = word_map['<unk>']
            scores[:, unk_token_index] -= 0.1

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_word_inds = top_k_words // vocab_size
            next_word_inds = top_k_words % vocab_size
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1).to(device)

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)

            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]].to(device)
            c1 = c1[prev_word_inds[incomplete_inds]].to(device)
            h2 = h2[prev_word_inds[incomplete_inds]].to(device)
            c2 = c2[prev_word_inds[incomplete_inds]].to(device)
            cluster_features_mean = cluster_features_mean[prev_word_inds[incomplete_inds]].to(device)
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1).to(device)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1).to(device)

            if step > 50:
                break
            step += 1

        hypotheses = []
        for i in range(len(complete_seqs)):
            seq = complete_seqs[i]
            hypothesis = [rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            hypothesis = ' '.join(hypothesis)
            score = complete_seqs_scores[i].item()
            hypotheses.append((hypothesis, score))
        
        return hypotheses

    all_captions = []
    for cluster in range(n_clusters):
        cluster_features = features[cluster_labels == cluster]
        partial_captions = generate_partial_caption(cluster_features, k, length_penalty)
        all_captions.append(partial_captions)

    def select_best_caption(captions):
        def descriptive_score(caption):
            doc = nlp(caption[0])
            num_descriptive_words = sum(1 for token in doc if token.pos_ in {'NOUN'})
            num_attributes = sum(0.5 for token in doc if token.pos_ in {'ADJ'})
            normalized_score = caption[1] / (1 + len(caption[0].split()))
            return normalized_score + num_descriptive_words + num_attributes
        
        return max(captions, key=descriptive_score)[0]

    best_captions = [select_best_caption(captions) for captions in all_captions]
    best_captions = [caption.capitalize() for caption in best_captions]
    return best_captions


def process_image(image_features, image_name):
    results = []
    # Ensure the decoder runs on CPU
    decoder.to("cpu")
    image_features = torch.FloatTensor(image_features).unsqueeze(0).to("cpu")
    captions = generate_caption(image_features)
    for j, caption in enumerate(captions):
        results.append({"image_name": image_name, "comment_number": j, "comment": caption + "."})
    return results


def get_processed_images(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep='|')
        if not df.empty:
            return set(df['image_name'])
    return set()

def generate_captions_for_images(npy_file_path, output_csv_path, batch_size=64):
    while True:
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

        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            for i in range(start_index, len(image_features_list), batch_size):
                batch_futures = {executor.submit(process_image, image_features_list[j], image_names[j]): j for j in range(i, min(i + batch_size, len(image_features_list)))}
                for future in tqdm(as_completed(batch_futures), total=len(batch_futures)):
                    try:
                        batch_results = future.result()
                        df_batch = pd.DataFrame(batch_results)
                        df_batch.to_csv(output_csv_path, mode='a', header=False, index=False, sep='|')
                    except Exception as e:
                        logger.error(f"Error processing image {image_names[batch_futures[future]]}: {e}")

        # Check if all images have been processed
        processed_images = get_processed_images(output_csv_path)
        if len(processed_images) == len(image_names):
            logger.info("All images processed. Exiting loop.")
            break

        logger.info("There are still images that are not captioned. Continuing my quest...")

if __name__ == '__main__':
    # Set multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn')

    npy_file_path = '../shared_data/flickr30k_name_features.npy'
    output_csv_path = '../shared_data/output_captions.csv'

    generate_captions_for_images(npy_file_path, output_csv_path)
