from typing import Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    M2M100Tokenizer,
    M2M100ForConditionalGeneration,
)


def load_model_and_tokenizer(
    model_name_or_path: str,
    local_model_path: str = "",
    fp16: bool = True,
    use_m2m100: bool = True,
    train_from_scratch: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer.

    Args:
        model_name_or_path: Model name or path
        local_model_path: Local path to model (if available)
        fp16: Whether to use fp16
        use_m2m100: Whether to use M2M100 specific loading
        train_from_scratch: If True, initialize model with random weights from config
                           instead of loading pretrained weights

    Note: output_attentions should be specified at generate() time,
    not during model loading, to avoid warnings.
    """
    model_path = local_model_path or model_name_or_path

    if use_m2m100 and "m2m100" in model_name_or_path.lower():
        # Use M2M100 specific loading
        tokenizer = M2M100Tokenizer.from_pretrained(model_path, cache_dir="model")

        if train_from_scratch:
            # Load config only and initialize with random weights
            config = AutoConfig.from_pretrained(model_path, cache_dir="model")
            model = M2M100ForConditionalGeneration(config)
            print(
                f"Initialized M2M100 model from scratch with config from {model_path}"
            )
        else:
            # Load pretrained weights
            model = M2M100ForConditionalGeneration.from_pretrained(
                model_path, cache_dir="model"
            )
            print(f"Loaded pretrained M2M100 model from {model_path}")

        if fp16 and torch.cuda.is_available():
            model = model.half()
    else:
        # Use generic AutoModel loading
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        if train_from_scratch:
            # Initialize with random weights
            model = AutoModelForSeq2SeqLM.from_config(config)
            print(f"Initialized model from scratch with config from {model_path}")
        else:
            # Load pretrained weights
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config)
            print(f"Loaded pretrained model from {model_path}")

        if fp16 and torch.cuda.is_available():
            model = model.half()

    return model, tokenizer


def set_m2m_langs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    src_lang: str,
    tgt_lang: str,
) -> None:
    """Set source and target languages for M2M100 and similar models."""
    # Set source and target language attributes
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = tgt_lang

    # Call the method to activate language-specific tokenization (if available)
    if hasattr(tokenizer, "set_src_lang_special_tokens"):
        tokenizer.set_src_lang_special_tokens(src_lang)

    # Set forced BOS token for decoder
    if hasattr(tokenizer, "get_lang_id"):
        lang_id = tokenizer.get_lang_id(tgt_lang)
        if hasattr(model.config, "forced_bos_token_id"):
            model.config.forced_bos_token_id = lang_id
