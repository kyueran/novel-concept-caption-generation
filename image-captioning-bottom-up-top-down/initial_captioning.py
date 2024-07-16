import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json
import os
import h5py

# Parameters
data_folder = 'final_dataset'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint_file = 'BEST_34checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint

word_map_file = 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cpu")  # sets device to CPU
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
torch.nn.Module.dump_patches = True
checkpoint = torch.load(checkpoint_file, map_location=device)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()

# Load word map (word2ix)
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

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

if __name__ == '__main__':
    # Load image features from the first entry in the HDF5 file
    '''
    h5_file_path = os.path.join(data_folder, 'val36.hdf5')  # Assuming validation set
    with h5py.File(h5_file_path, 'r') as h:
        image_features = torch.FloatTensor(h['image_features'][0])  # Load the first image's features
    '''

    npy_file_path = '../roi_features.npy'  # Path to the numpy file
    image_features = torch.FloatTensor(np.load(npy_file_path))
    # Reshape image_features to match the expected input shape
    image_features = image_features.unsqueeze(0)  # Add batch dimension

    # Generate caption
    caption = generate_caption(image_features, beam_size=5)
    print("Generated Caption:", caption)
