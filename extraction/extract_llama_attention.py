"""
Extract attention matrices from Llama-3.1-8B-Instruct for translation tasks.

For each sentence in the WMT test set, uses a prompt-based approach to
translate to all target languages and extracts clipped attention matrices
(attention from translated tokens to source sentence tokens only).

Usage:
    python -m extraction.extract_llama_attention
"""

import os
import pickle
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map, dispatch_model

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    set_seed_everywhere,
    get_text_token_positions,
    accumulate_attention,
    extract_translation_with_positions,
    get_all_lang,
)


def get_clip_acc_atten(tokenizer, model, src_lang="English", tgt_lang="French",
                       text="How are you?", verbose=False):
    """Generate translation and return clipped attention matrices."""
    prompt = (
        f"Translate the following sentence from {src_lang} to {tgt_lang}. "
        f"Only reply with the translated sentence, strictly using the format "
        f"'<START> translation <END>'. "
        f" Sentence to translate: <<{text}>> Here is the correct translation: <START> "
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    prompt_text_length = len(prompt)
    prompt_token_length = len(inputs["input_ids"][0])

    start_position, end_position = get_text_token_positions(prompt, text, tokenizer)
    if start_position is None:
        return (None, None, None,
                torch.ones((NUM_LAYERS, 1, 1, 1)),
                torch.ones((NUM_LAYERS, 1, 1, 1)),
                torch.ones((NUM_LAYERS, 1, 1, 1)))

    text_tokens = inputs["input_ids"][0][start_position + 1 : end_position + 1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            return_dict_in_generate=True,
            output_attentions=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    translation, start_id, end_id = extract_translation_with_positions(
        generated_text[prompt_text_length - 8 :], tokenizer
    )
    print(f"{tgt_lang} Translation: {translation}", start_id, end_id, len(outputs.sequences[0]))

    if translation is None:
        return (None, None, None,
                torch.ones((NUM_LAYERS, 1, 1, 1)),
                torch.ones((NUM_LAYERS, 1, 1, 1)),
                torch.ones((NUM_LAYERS, 1, 1, 1)))

    attention = outputs.attentions
    attention_matrices = {}
    prompt_attention_matrices = {}

    for step, step_attention in enumerate(attention):
        if step == 0:
            for layer, layer_attention in enumerate(step_attention):
                if layer not in prompt_attention_matrices:
                    prompt_attention_matrices[layer] = []
                prompt_attention_matrices[layer].append(layer_attention)
            continue
        for layer, layer_attention in enumerate(step_attention):
            prompt_attention = layer_attention[:, :, -1:, :]
            if layer not in attention_matrices:
                attention_matrices[layer] = []
            attention_matrices[layer].append(prompt_attention)

    # Build clip attention (only attention to prompt tokens)
    all_layers_attention = []
    for layer, attentions in attention_matrices.items():
        clip_attentions = [att[:, :, :, :prompt_token_length] for att in attentions]
        concatenated_attention = torch.cat(clip_attentions, dim=2)
        all_layers_attention.append(concatenated_attention)
    clip_attention_matrix = torch.cat(all_layers_attention, dim=0)

    # Build accumulated attention
    final_attention_per_layer = []
    for layer, attentions in attention_matrices.items():
        accumulated_attention = accumulate_attention(attentions, prompt_token_length)
        final_attention_per_layer.append(accumulated_attention)
    acc_attention_matrix = torch.cat(final_attention_per_layer, dim=0)

    sel_clip_attention_matrix = clip_attention_matrix[
        :, :, : (end_id - start_id), start_position + 1 : end_position + 1
    ]
    sel_acc_attention_matrix = acc_attention_matrix[
        :, :, : (end_id - start_id), start_position + 1 : end_position + 1
    ]

    prompt_attention_matrix = torch.cat(
        [torch.stack(att, dim=0) for att in prompt_attention_matrices.values()], dim=0
    )

    translation_tokens = tokenizer(translation, return_tensors="pt")["input_ids"][0]
    input_tokens = [tokenizer.decode([token]) for token in text_tokens]
    output_tokens = [tokenizer.decode([token]) for token in translation_tokens[1:]]

    return (input_tokens, output_tokens, generated_text,
            sel_clip_attention_matrix, sel_acc_attention_matrix, prompt_attention_matrix)


def get_atten_for_lang_list(one_text, src_lang, tgt_lang_list, tokenizer, model):
    """Obtain attention matrix for each target language."""
    atten_matrix_dict = {}
    tgt_lang_dict, _ = get_all_lang()

    for idx, tgt_lang in enumerate(tgt_lang_list):
        t0 = time.time()
        (input_tokens, output_tokens, generated_text,
         clip_attention_matrix, acc_attention_matrix, prompt_attention_matrix,
        ) = get_clip_acc_atten(
            tokenizer, model, src_lang, tgt_lang_dict[tgt_lang], one_text
        )

        t1 = time.time()
        print(f"Time: {t1 - t0:.1f}s")
        print("=" * 50)

        atten_matrix_dict[tgt_lang] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generated_answer": generated_text,
            "attention_matrix": clip_attention_matrix.half().cpu().numpy(),
        }
    return atten_matrix_dict


def eval_all():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, output_attentions=True, torch_dtype=torch.bfloat16,
    )
    print(f"Available CUDA devices: {torch.cuda.device_count()}")

    device_map = infer_auto_device_map(
        model,
        max_memory={i: "24GB" for i in range(max(1, torch.cuda.device_count()))},
        no_split_module_classes=["LlamaDecoderLayer"],
    )
    model = dispatch_model(model, device_map=device_map)
    model.eval()

    data_dir = "./data"
    with open(os.path.join(data_dir, "wmt.test.fr-en.en"), "r") as f:
        text_list = f.readlines()
    text_list = [text.strip() for text in text_list]

    _, tgt_lang_list = get_all_lang()
    print("Number of target languages:", len(tgt_lang_list))

    for idx, each_text in tqdm(enumerate(text_list)):
        result = get_atten_for_lang_list(
            each_text, "English", tgt_lang_list, tokenizer, model
        )
        with open(f"{OUTPUT_DIR}/atten_matrix/{idx}.pkl", "wb") as f:
            pickle.dump(result, f)


if __name__ == "__main__":
    set_seed_everywhere(17)

    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    OUTPUT_DIR = "Meta-Llama-3.1-8B-Instruct"
    NUM_LAYERS = 32

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/atten_matrix", exist_ok=True)

    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {DEVICE}")

    t0 = time.time()
    eval_all()
    print(f"Total time: {time.time() - t0:.1f}s")
