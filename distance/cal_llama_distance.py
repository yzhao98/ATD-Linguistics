"""
Compute pairwise ATD (Attention Transport Distance) scores for Llama-3.1-8B.

For each sentence, computes the Wasserstein-2 distance between clipped
attention distributions of all language pairs at each layer.

Usage:
    python -m distance.cal_llama_distance
"""

import os
import pickle

import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_all_lang, compute_w2_distance

NUM_LAYERS = 32


if __name__ == "__main__":
    data_dir = "./data"
    with open(os.path.join(data_dir, "wmt.test.fr-en.en"), "r") as f:
        text_list = f.readlines()
    text_list = [text.strip() for text in text_list]

    lang_dict, sel_lang_list = get_all_lang()
    OUT_DIR = "Meta-Llama-3.1-8B-Instruct/atten_rollout"
    all_distance_dict = {}
    save_interval = 100

    for idx, origin_text in tqdm(enumerate(text_list)):
        filename = f"{OUT_DIR}/{idx}.pkl"
        with open(filename, "rb") as f:
            data = pickle.load(f)

        clip_atten_dict = {}
        for tgt_lang in data.keys():
            if tgt_lang not in sel_lang_list:
                continue
            # clip_attention_matrix: [layer, head, output, input] -> mean over head and output -> [layer, input]
            clip_atten_dict[tgt_lang] = data[tgt_lang]["clip_attention_matrix"].mean(1)

        # Compute pairwise W2 distance at each layer
        distance_matrix = np.zeros((len(sel_lang_list), len(sel_lang_list), NUM_LAYERS))
        for i, lang1 in enumerate(data.keys()):
            for j, lang2 in enumerate(data.keys()):
                if i >= j:
                    continue
                for n_layer in range(NUM_LAYERS):
                    ot_distance = compute_w2_distance(
                        clip_atten_dict, lang1, lang2, n_layer
                    )
                    distance_matrix[i][j][n_layer] = ot_distance
                    distance_matrix[j][i][n_layer] = ot_distance

        all_distance_dict[idx] = distance_matrix

        if idx % save_interval == 0:
            with open("results_llama3/all_llama3_distance_dict_fixed_clip_layer.pkl", "wb") as f:
                pickle.dump(all_distance_dict, f)

    # Final save
    with open("results_llama3/all_llama3_distance_dict_fixed_clip_layer.pkl", "wb") as f:
        pickle.dump(all_distance_dict, f)
