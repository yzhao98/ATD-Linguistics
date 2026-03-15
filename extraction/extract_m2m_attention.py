"""
Extract cross-attention matrices from M2M-100 for all language pairs.

For each sentence in the WMT test set, translates to all target languages
and saves the cross-attention matrices as pickle files.

Usage:
    python -m extraction.extract_m2m_attention
"""

import os
import pickle
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import set_seed_everywhere, get_all_lang


def get_atten_for_lang_list(one_text, src_lang, tgt_lang_list, tokenizer, model, atten_matrix_dict):
    """Generate translations and extract cross-attention for each target language."""
    for idx, tgt_lang in enumerate(tgt_lang_list):
        tokenizer.src_lang = src_lang
        model.config.forced_bos_token_id = tokenizer.get_lang_id(tgt_lang)
        t0 = time.time()

        encoded = tokenizer(one_text, return_tensors="pt").to(DEVICE)
        decoder_input_ids = torch.full(
            (1, 1), model.config.forced_bos_token_id, dtype=torch.long, device=DEVICE
        )

        outputs = model.generate(
            **encoded,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            return_dict_in_generate=True,
        )

        translated_tokens = outputs.sequences
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        print(f"Original text: {one_text}")
        print(f"Translated text: {translated_text}")
        print(tgt_lang, len(translated_tokens[0]))

        all_cross_attentions = []
        for i, attention in enumerate(outputs.cross_attentions):
            layer_stacked_attentions = torch.stack(attention, dim=1)
            all_cross_attentions.append(layer_stacked_attentions)
        concatenated_cross_attentions = torch.cat(all_cross_attentions, dim=3)

        t1 = time.time()
        print(f"Time: {t1 - t0:.1f}s")
        print("=" * 50)

        atten_matrix_dict[tgt_lang] = {
            "generated_answer": translated_text,
            "input_tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])[1:-1],
            "output_tokens": tokenizer.convert_ids_to_tokens(translated_tokens[0])[2:-1],
            "attention_matrix": concatenated_cross_attentions.cpu().numpy()[0, :, :, 2:, 1:-1],
        }
    return atten_matrix_dict


def eval_all():
    model_name = "facebook/m2m100_1.2B"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name, cache_dir="model")
    model = M2M100ForConditionalGeneration.from_pretrained(
        model_name, output_attentions=True, cache_dir="model"
    ).to(DEVICE)
    model.config.num_beams = 5
    model.eval()

    data_dir = "./data"
    with open(os.path.join(data_dir, "wmt.test.fr-en.en"), "r") as f:
        text_list = f.readlines()
    text_list = [text.strip() for text in text_list]

    tgt_lang_dict, tgt_lang_list = get_all_lang()
    print("Number of target languages:", len(tgt_lang_list))

    for idx, each_text in tqdm(enumerate(text_list)):
        last_result = {}
        output_path = f"{OUTPUT_DIR}/atten_matrix_100/{idx}.pkl"
        try:
            with open(output_path, "rb") as f:
                last_result = pickle.load(f)
            print(f"Loaded {idx} from file.")
        except FileNotFoundError:
            print(f"File {idx} not found, generating new one.")

        result = get_atten_for_lang_list(
            each_text, "en", tgt_lang_list, tokenizer, model, last_result
        )
        with open(output_path, "wb") as f:
            pickle.dump(result, f)


if __name__ == "__main__":
    set_seed_everywhere(17)

    OUTPUT_DIR = "models--facebook--m2m100_1.2B_5beams"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/atten_matrix_100", exist_ok=True)

    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    NUM_LAYERS = 24

    print(f"Device: {DEVICE}")
    eval_all()
