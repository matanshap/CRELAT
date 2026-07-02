
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.utils import tokenize
import json
import os
import random
import numpy as np
from scipy import spatial
import pandas as pd
import torch

def read_corpus(corpus):
    for doc in corpus:
        yield list(tokenize(doc))

def _train_forward_only(corpus_list, vector_size, window_size, epochs=10, negative=10, min_count=1):
    """
    Train Word2Vec with forward-only context: [pos : pos+2*window] instead of [pos-window : pos+window].
    Uses skip-gram with negative sampling.
    """
    from collections import Counter
    counts = Counter(corpus_list)
    vocab = [w for w, c in counts.items() if c >= min_count]
    if not vocab:
        return {"vectors": {}}
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    # Negative sampling table (weighted by count^0.75)
    counts_arr = np.array([counts[w] ** 0.75 for w in vocab])
    neg_table = np.repeat(np.arange(V), np.maximum(1, (counts_arr / counts_arr.sum() * 1e8).astype(int)))
    neg_table = neg_table[: min(int(1e8), len(neg_table))]

    # Initialize embeddings
    W = np.random.randn(V, vector_size) * 0.01
    W_out = np.random.randn(V, vector_size) * 0.01

    # Build forward-only training pairs (target, context) where context is in [pos+1, pos+2*window]
    pairs = []
    tokens = corpus_list
    for i in range(len(tokens)):
        if tokens[i] not in w2i:
            continue
        idx_i = w2i[tokens[i]]
        for j in range(i + 1, min(i + 2 * window_size + 1, len(tokens))):
            if tokens[j] not in w2i:
                continue
            idx_j = w2i[tokens[j]]
            pairs.append((idx_i, idx_j))

    if not pairs:
        return {"vectors": {w: W[w2i[w]].tolist() for w in vocab}}

    # Training
    lr = 0.025
    for ep in range(epochs):
        lr = max(0.0001, lr)
        random.shuffle(pairs)
        for idx_i, idx_j in pairs:
            negs = list(np.random.choice(neg_table, size=min(negative, len(neg_table)), replace=True))
            negs = [n for n in negs if n != idx_j and n != idx_i][:negative]

            v_w = W[idx_i]
            u_c = W_out[idx_j]

            # Positive pair gradient
            score = np.dot(v_w, u_c)
            sig = 1 / (1 + np.exp(-np.clip(score, -500, 500)))
            grad = sig - 1
            W[idx_i] -= lr * grad * u_c
            W_out[idx_j] -= lr * grad * v_w

            # Negative pairs
            for n in negs:
                u_n = W_out[n]
                score_n = np.dot(v_w, u_n)
                sig_n = 1 / (1 + np.exp(-np.clip(score_n, -500, 500)))
                grad_n = sig_n
                W[idx_i] -= lr * grad_n * u_n
                W_out[n] -= lr * grad_n * v_w

        lr *= 0.995

    return {"vectors": {w: W[w2i[w]].tolist() for w in vocab}}

def remove_duplicates(data):
    seen = set()
    unique_data = []
    for sublist in data:
        # Sort the first two elements to ignore order
        pair = tuple(sorted(sublist[:2]))
        if pair not in seen:
            seen.add(pair)
            unique_data.append(sublist)
    return unique_data

def TrainW2VModel(book_name, corpus_list, vector_size, window_size, output_path, forward_only=False):
    """
    Train Word2Vec on corpus_list. If forward_only=True, use context [pos : pos+2*window]
    instead of symmetric [pos-window : pos+window].
    """
    output_text_file = output_path + book_name + "_w2v.txt"
    dir_path = os.path.dirname(output_text_file)
    # create the directory if it does not exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if forward_only:
        v = _train_forward_only(corpus_list, vector_size, window_size, epochs=10, negative=10)
    else:
        corpus = list(read_corpus(corpus_list))
        new_list = [[]]
        for i in corpus_list:
            new_list[0].append(i)
        fine_tuned_model = Word2Vec(
            new_list, vector_size=vector_size, window=window_size,
            min_count=1, workers=4, epochs=10, negative=10
        )
        fine_tuned_model.wv.save_word2vec_format(output_text_file, binary=False)
        f = open(output_text_file)
        v = {"vectors": {}}
        for line in f:
            w, n = line.split(" ", 1)
            v["vectors"][w] = list(map(float, n.split()))
        f.close()

    if forward_only:
        with open(output_text_file, "w") as f:
            for w, vec in v["vectors"].items():
                f.write(w + " " + " ".join(str(x) for x in vec) + "\n")

    # Save to a JSON file
    # Could make this an optional argument to specify output file
    with open(output_text_file[:-4]+".json", "w") as out:
        #json.dump(v, out)
        json.dump(v, out)


    print("(TrainW2VModel) Done")
    return v





def GenW2V(entities,vectors):
    # All possible pairs in List
    all_pairs = [(a, b) for idx, a in enumerate(entities) for b in entities[idx + 1:]]
    for i in range(0,len(all_pairs)):
      all_pairs[i]=list(all_pairs[i])
    i = 0
    if isinstance(all_pairs[0][0],list):
        first_entity = all_pairs[0][0][0]
    else:
        first_entity = all_pairs[0][0]
    #print(vectors['vectors'])
    if isinstance(vectors['vectors'][first_entity], torch.Tensor):
        vectors['vectors'] = tensors_to_lists(vectors['vectors'])
    # compute cosine similarity

    while i < len(all_pairs):
        if len(all_pairs[i][0][0])>1:
            first_in_pair = all_pairs[i][0][0]
            sec_in_pair = all_pairs[i][1][0]
        else:
            first_in_pair = all_pairs[i][0]
            sec_in_pair = all_pairs[i][1]
        if ((not (first_in_pair in vectors['vectors'])) or (not (sec_in_pair in vectors['vectors']))):
            del all_pairs[i]
            continue
        sim1 = 1 - spatial.distance.cosine(vectors['vectors'][first_in_pair], vectors['vectors'][sec_in_pair])
        all_pairs[i].append(sim1)
        i += 1

    all_pairs.sort(key=lambda x: x[2])
    all_pairs.reverse()
    for i in range(len(all_pairs)):
        if len(all_pairs[i][0][0])>1:
            all_pairs[i][0] = all_pairs[i][0][0]
            all_pairs[i][1] = all_pairs[i][1][0]
        else:
            all_pairs[i][0] = all_pairs[i][0]
            all_pairs[i][1] = all_pairs[i][1]

    all_pairs = remove_duplicates(all_pairs)
    print("(GenW2V) Done")
    return all_pairs

def tensors_to_lists(tensor_dict):
    list_dict = {}
    for key, tensor in tensor_dict.items():
        # Convert PyTorch tensor to list
        list_dict[key] = tensor.tolist()
    return list_dict



