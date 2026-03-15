"""
Compute pairwise ATD (Attention Transport Distance) scores for M2M-100.

For each sentence, computes the Wasserstein-2 distance between attention
distributions of all language pairs at each layer.

Usage:
    python -m distance.cal_m2m_distance
"""

import os
import pickle

import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_all_lang, compute_w2_distance

NUM_LAYERS = 24

if __name__ == "__main__":
    data_dir = "./data"
    with open(os.path.join(data_dir, "wmt.test.fr-en.en"), "r") as f:
        text_list = f.readlines()
    text_list = [text.strip() for text in text_list]

    lang_dict, sel_lang_list = get_all_lang()
    OUT_DIR = "models--facebook--m2m100_1.2B_5beams/atten_matrix_100"
    all_distance_dict = {}
    save_interval = 100

    for idx, origin_text in tqdm(enumerate(text_list)):
        filename = f"{OUT_DIR}/{idx}.pkl"
        with open(filename, "rb") as f:
            data = pickle.load(f)

        # Compute accumulated attention per language
        input_accumulated_atten_dict = {}
        for tgt_lang in data.keys():
            if tgt_lang not in sel_lang_list:
                continue
            # Mean over heads and output tokens: [layer, head, output, input] -> [layer, input]
            input_accumulated_atten = data[tgt_lang]["attention_matrix"].mean(1).mean(1)
            input_accumulated_atten_dict[tgt_lang] = input_accumulated_atten

        # Compute pairwise W2 distance at each layer
        distance_matrix = np.zeros((len(sel_lang_list), len(sel_lang_list), NUM_LAYERS))
        for i, lang1 in enumerate(lang_dict.keys()):
            for j, lang2 in enumerate(lang_dict.keys()):
                if i >= j:
                    continue
                if lang1 not in input_accumulated_atten_dict or lang2 not in input_accumulated_atten_dict:
                    continue
                for n_layer in range(NUM_LAYERS):
                    ot_distance = compute_w2_distance(
                        input_accumulated_atten_dict, lang1, lang2, n_layer
                    )
                    distance_matrix[i][j][n_layer] = ot_distance
                    distance_matrix[j][i][n_layer] = ot_distance

        all_distance_dict[idx] = distance_matrix

        if idx % save_interval == 0:
            with open("results_m2m/all_m2m_distance_dict_fixed_ot.pkl", "wb") as f:
                pickle.dump(all_distance_dict, f)

    # Final save
    with open("results_m2m/all_m2m_distance_dict_fixed_ot.pkl", "wb") as f:
        pickle.dump(all_distance_dict, f)
