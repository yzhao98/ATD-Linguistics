"""
Evaluate Llama-3.1-8B translation quality using GPT-4o.

Scores each translation as: yes (1.0), almost (0.5), no (0.0),
or -1 (extraction failed) / -2 (generation returned None).

Usage:
    export OPENAI_API_KEY="your-key-here"
    python -m evaluation.eval_quality_llama
"""

import os
import pickle
import re

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_all_lang, eval_response_by_llm

# Set API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

data_dir = "./data"
with open(os.path.join(data_dir, "wmt.test.fr-en.en"), "r") as f:
    text_list = f.readlines()
text_list = [text.strip() for text in text_list]

OUT_DIR = "Meta-Llama-3.1-8B-Instruct/atten_matrix"
RESULTS_DIR = "results_llama3"
tgt_lang_dict, _ = get_all_lang()


def save_trans_results_to_dict():
    """Extract translations from attention pickle files into a single dict."""
    result_dict = {}
    for i, origin_text in enumerate(text_list):
        filename = f"{OUT_DIR}/{i}.pkl"
        with open(filename, "rb") as f:
            data = pickle.load(f)
        result_dict[i] = {"original_text": origin_text}
        for tgt_lang in data.keys():
            generated_text = data[tgt_lang]["generated_answer"]
            if generated_text is None:
                translation = "None."
            else:
                match = re.search(r"<START>(.*?)<END>", generated_text[155:])
                translation = match.group(1).strip() if match else "Wrong."
            result_dict[i][tgt_lang] = translation

    with open(f"{RESULTS_DIR}/llama3_trans_result_dict.pkl", "wb") as f:
        pickle.dump(result_dict, f)


def eval_trans_results(start=0, end=3003):
    """Evaluate translation quality for a range of sentences."""
    columns = list(tgt_lang_dict.keys())
    num_rows = len(text_list)
    result_df = pd.DataFrame(np.zeros((num_rows, len(columns))), columns=columns)

    with open(f"{RESULTS_DIR}/llama3_trans_result_dict.pkl", "rb") as f:
        result_dict = pickle.load(f)

    for i, origin_text in tqdm(enumerate(text_list)):
        if i not in range(start, end):
            continue
        original_text = result_dict[i]["original_text"]
        for tgt_lang in result_dict[i].keys():
            if tgt_lang == "original_text":
                continue
            translated_text = result_dict[i][tgt_lang]

            if translated_text == "Wrong.":
                result_df.at[i, tgt_lang] = -1
                continue
            elif translated_text == "None.":
                result_df.at[i, tgt_lang] = -2
                continue

            eval_result = eval_response_by_llm(original_text, translated_text, tgt_lang)
            if eval_result == "yes":
                result_df.at[i, tgt_lang] = 1
            elif eval_result == "almost":
                result_df.at[i, tgt_lang] = 0.5
            elif eval_result == "no":
                result_df.at[i, tgt_lang] = 0
            else:
                result_df.at[i, tgt_lang] = np.nan

        result_df.to_csv(
            f"{RESULTS_DIR}/llama3_trans_llm_eval_{start}_{end}.csv", index=False
        )


if __name__ == "__main__":
    # Step 1: Extract translations (run once)
    # save_trans_results_to_dict()

    # Step 2: Evaluate quality
    eval_trans_results(start=0, end=3003)
