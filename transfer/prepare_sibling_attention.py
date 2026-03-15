#!/usr/bin/env python3
"""
Prepare sibling attention references - V3 with optimized processing order.

Key improvements:
- Process all samples for each language (minimize model switching: only 2 times total!)
- Per-sample storage to avoid OOM
- Immediate CPU transfer to free GPU memory
- Validation to ensure attention extraction works correctly
"""

import argparse
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transfer.datasets import ParallelTextDataset
from transfer.models import load_model_and_tokenizer, set_m2m_langs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--src_lang", type=str, required=True)
    p.add_argument(
        "--target_lang", type=str, required=True, help="Target language for training"
    )
    p.add_argument(
        "--sibling_langs",
        type=str,
        nargs="+",
        required=True,
        help="Sibling languages to use as reference (e.g., hi ur)",
    )
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save individual attention files",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of samples to process (None=all samples)",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument(
        "--num_beams", type=int, default=1, help="Number of beams for generation"
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Max tokens to generate (None=no limit)",
    )
    p.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start from this sample index (for resuming)",
    )
    p.add_argument(
        "--reduced_only",
        action="store_true",
        help=(
            "If set, save only per-layer averaged attention over heads and all steps"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Preparing sibling attention references (V3 - Optimized)")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Source language: {args.src_lang}")
    print(f"Target language (training): {args.target_lang}")
    print(f"Sibling languages (reference): {args.sibling_langs}")
    print(f"Train file: {args.train_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num beams: {args.num_beams}")
    print(
        f"Max new tokens: {args.max_new_tokens if args.max_new_tokens else 'no limit'}"
    )
    print(f"Start index: {args.start_idx}")
    print("=" * 70)
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, local_model_path="", fp16=False
    )
    print(f"Moving model to {device}...")
    model.to(device)
    model.eval()

    # Load dataset
    print(f"\nLoading dataset from {args.train_file}...")
    dataset = ParallelTextDataset(args.train_file)

    if args.max_samples is not None:
        dataset.examples = dataset.examples[: args.max_samples]
        print(f"Limited to {args.max_samples} samples")

    print(f"Total samples: {len(dataset)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    # Adjust start index
    start_idx = max(args.start_idx, 0)
    print(f"\nProcessing samples from {start_idx} to {len(dataset)}...")
    print(f"Batch size: {args.batch_size}")
    print(f"\n🚀 Strategy: Process all samples for EACH language separately")
    print(f"   This minimizes model switching to only {len(args.sibling_langs)} times!")

    # Track validation status
    attention_validated = False

    # OUTER LOOP: Process by language (minimize model switching!)
    for lang_idx, sib_lang in enumerate(args.sibling_langs):
        print(f"\n{'='*70}")
        print(f"Language {lang_idx+1}/{len(args.sibling_langs)}: {sib_lang}")
        print(f"{'='*70}")

        # Set language ONCE for all batches of this language
        set_m2m_langs(model, tokenizer, args.src_lang, sib_lang)
        print(f"✓ Model set to {args.src_lang} -> {sib_lang}")

        # Check which samples need this language
        samples_needing_this_lang = []
        for sample_idx in range(start_idx, len(dataset)):
            sample_file = output_dir / f"{sample_idx}.pkl"
            needs_processing = True

            if sample_file.exists():
                with open(sample_file, "rb") as f:
                    sample_data = pickle.load(f)
                if sib_lang in sample_data:
                    needs_processing = False

            if needs_processing:
                samples_needing_this_lang.append(sample_idx)

        if not samples_needing_this_lang:
            print(f"✓ All samples already have {sib_lang}, skipping")
            continue

        print(f"Processing {len(samples_needing_this_lang)} samples for {sib_lang}...")

        # INNER LOOP: Process all samples for this language in batches
        processed_count = 0
        for batch_start_idx in tqdm(
            range(0, len(samples_needing_this_lang), args.batch_size),
            desc=f"{sib_lang}",
        ):
            batch_end_idx = min(
                batch_start_idx + args.batch_size, len(samples_needing_this_lang)
            )
            batch_sample_indices = samples_needing_this_lang[
                batch_start_idx:batch_end_idx
            ]

            # Get batch of source texts
            batch_src_texts = [dataset.examples[i][0] for i in batch_sample_indices]

            # Tokenize batch
            inputs = tokenizer(
                batch_src_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)

            # Generate with attention
            with torch.no_grad():
                # Align with eval_m2m style: provide decoder_input_ids starting with forced BOS
                batch_bsz = inputs["input_ids"].size(0)
                bos_id = int(getattr(model.config, "forced_bos_token_id", -1))
                decoder_input_ids = None
                if bos_id >= 0:
                    decoder_input_ids = torch.full(
                        (batch_bsz, 1), bos_id, dtype=torch.long, device=device
                    )
                gen_kwargs = {
                    **inputs,
                    "do_sample": False,
                    "num_beams": args.num_beams,
                    "early_stopping": True,
                    "length_penalty": 0.9,
                    "output_attentions": True,
                    "return_dict_in_generate": True,
                    "use_cache": False,
                    "eos_token_id": tokenizer.eos_token_id,
                    "no_repeat_ngram_size": 3,
                    "repetition_penalty": 1.15,
                }
                if decoder_input_ids is not None:
                    gen_kwargs["decoder_input_ids"] = decoder_input_ids
                # Enforce max length
                _mn = args.max_new_tokens if args.max_new_tokens is not None else 64
                gen_kwargs["max_new_tokens"] = _mn
                gen_kwargs["max_length"] = 1 + _mn  # BOS(1) + new tokens

                outputs = model.generate(**gen_kwargs)

            # Validate attention extraction on first batch
            if not attention_validated:
                if hasattr(outputs, "cross_attentions") and outputs.cross_attentions:
                    num_steps = len(outputs.cross_attentions)
                    num_layers = (
                        len(outputs.cross_attentions[0])
                        if outputs.cross_attentions
                        else 0
                    )
                    print(f"\n✓ Attention extraction verified!")
                    print(f"  - Generation steps: {num_steps}")
                    print(f"  - Layers per step: {num_layers}")
                    if num_layers > 0:
                        sample_shape = outputs.cross_attentions[0][0].shape
                        print(
                            f"outputs.cross_attentions: {len(outputs.cross_attentions)}, {len(outputs.cross_attentions[0])}"
                        )
                        print(f"  - Attention shape: {sample_shape}")
                    attention_validated = True
                else:
                    print(f"\n✗ ERROR: No cross_attentions found!")
                    print(f"  This should not happen. Check generate() parameters.")
                    print(f"  outputs has attributes: {dir(outputs)}")
                    raise RuntimeError("Failed to extract cross_attentions")

            # Extract cross-attentions for ALL generation steps and move to CPU
            if hasattr(outputs, "cross_attentions") and outputs.cross_attentions:
                # outputs.cross_attentions: List[num_steps] where each is a tuple of per-layer tensors [B,H,1,S]

                # Process each sample in the batch
                batch_size_actual = inputs["input_ids"].size(0)
                for sample_idx_in_batch in range(batch_size_actual):
                    global_sample_idx = batch_sample_indices[sample_idx_in_batch]

                    # Build steps x layers structure with tensors on CPU, each [1,H,1,S]
                    steps_layers = []
                    for step_attn in outputs.cross_attentions:
                        layer_list = []
                        for layer_attn in step_attn:
                            # layer_attn: [B, H, tgt_len(=1), src_len]
                            sample_step_attn = layer_attn[
                                sample_idx_in_batch : sample_idx_in_batch + 1, :, :, :
                            ].cpu()
                            layer_list.append(sample_step_attn)
                        steps_layers.append(layer_list)

                    # Load existing data for this sample (if any)
                    sample_file = output_dir / f"{global_sample_idx}.pkl"
                    if sample_file.exists():
                        with open(sample_file, "rb") as f:
                            sample_data = pickle.load(f)
                    else:
                        sample_data = {}

                    # Compute reduced per-layer averages using your [B,L,H,T,S] method
                    # 1) For each step, stack per-layer tensors on dim=1 → [1,L,H,1,S]
                    # 2) Concatenate steps on dim=3 → [1,L,H,T,S]
                    # 3) Mean over H and T → [1,L,S]
                    # 4) Split by layer → List[L] of [1,S]

                    # per-sample concat across steps -> [B,L,H,T,S], then average over H & T
                    all_cross_attentions = [
                        torch.stack(step, dim=1) for step in outputs.cross_attentions
                    ]  # each [B,L,H,1,S]
                    if all_cross_attentions:
                        concatenated = torch.cat(
                            all_cross_attentions, dim=3
                        )  # [B,L,H,T,S]
                        sample_full = concatenated[
                            sample_idx_in_batch : sample_idx_in_batch + 1
                        ]  # [1,L,H,T,S] -> [1, layer, head, output, input]
                        # print(f"sample_full shape: {sample_full.shape}")
                        reduced_all = sample_full.mean(dim=2).mean(
                            dim=2
                        )  # [1,L,S] -> [1, layer, S]
                        # print(f"reduced_all shape: {reduced_all.shape}")
                        per_layer_avg = [
                            reduced_all[:, li, :].contiguous().cpu()
                            for li in range(reduced_all.shape[1])
                        ]
                    else:
                        per_layer_avg = []

                    per_layer_avg_matrix = torch.cat(per_layer_avg, dim=0)
                    print(f"per_layer_avg_matrix: {per_layer_avg_matrix.shape}")
                    # Write outputs
                    if args.reduced_only:
                        sample_data = {
                            "attention_reduced": {
                                sib_lang: {"per_layer_avg": per_layer_avg_matrix}
                            }
                        }
                    else:
                        # Raw full steps
                        if "attention_data" not in sample_data:
                            sample_data["attention_data"] = {}
                        sample_data["attention_data"][sib_lang] = {
                            "cross_attentions": steps_layers
                        }
                        # And reduced
                        if "attention_reduced" not in sample_data:
                            sample_data["attention_reduced"] = {}
                        sample_data["attention_reduced"][sib_lang] = {
                            "per_layer_avg": per_layer_avg_matrix
                        }

                    # Save immediately
                    with open(sample_file, "wb") as f:
                        pickle.dump(sample_data, f)

                    processed_count += 1
            else:
                print(f"\n✗ WARNING: No cross_attentions in batch {batch_start_idx}")

            # Free GPU memory immediately after each batch
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Periodic cleanup every 100 batches
            if batch_start_idx % (100 * args.batch_size) == 0 and batch_start_idx > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"✓ Completed {sib_lang}: processed {processed_count} samples")

    print("\n" + "=" * 70)
    print("🎉 Done!")
    print(f"Saved attention for {len(args.sibling_langs)} languages to {output_dir}")
    print(f"File format: 0.pkl, 1.pkl, 2.pkl, ...")
    print(f"Each file contains: {{lang: [layer_tensors], ...}}")
    print("=" * 70)


if __name__ == "__main__":
    main()
