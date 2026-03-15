#!/usr/bin/env python3
"""
Training script with sibling attention penalty using precomputed attention references.
"""

import argparse
import csv
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, AdamW

from .attention import (
    attention_sibling_penalty_geomloss_batched,
    AttentionRefStore,
    cosine_penalty_to_ref,
    refdir_sinkhorn_penalty_to_ref,
)
from .datasets import ParallelTextDataset, DataCollatorSeq2Seq
from .models import load_model_and_tokenizer, set_m2m_langs
from .utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train with sibling attention penalty")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_model_path", type=str, default="")
    p.add_argument("--src_lang", type=str, required=True)
    p.add_argument("--tgt_lang", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--valid_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    # Attention penalty parameters
    p.add_argument(
        "--lambda_attn", type=float, default=0.0, help="Weight for attention penalty"
    )
    p.add_argument(
        "--lambda_sibling",
        type=float,
        default=0.0,
        help="Weight for sibling attention penalty",
    )
    p.add_argument(
        "--diag_sigma", type=float, default=3.0, help="Sigma for diagonal prior"
    )
    p.add_argument(
        "--sibling_penalty_type",
        type=str,
        default="geomloss",
        choices=["geomloss", "pot", "wasserstein_cdf"],
        help="Type of sibling penalty to use",
    )
    p.add_argument(
        "--sibling_attention_file",
        type=str,
        default="",
        help="Path to precomputed sibling attention file",
    )
    p.add_argument(
        "--geomloss_blur", type=float, default=0.05, help="Blur parameter for GeomLoss"
    )
    p.add_argument(
        "--layer_stride", type=int, default=1, help="Stride for layer sampling"
    )
    # External attention references
    p.add_argument(
        "--ref_attention_dir",
        type=str,
        default="",
        help="Directory with per-sentence attention pkl files (ALL steps)",
    )
    p.add_argument(
        "--ref_langs",
        type=str,
        nargs="*",
        default=[],
        help="Choose which reference languages to use (e.g., hi pa)",
    )
    p.add_argument(
        "--lambda_ref",
        type=float,
        default=0.0,
        help="Weight for external reference attention penalty (interpretation depends on --lambda_ref_mode)",
    )
    p.add_argument(
        "--lambda_ref_mode",
        type=str,
        default="fixed",
        choices=["fixed", "relative"],
        help="Lambda scaling mode: 'fixed' = constant weight, 'relative' = maintain fixed ratio to CE loss",
    )
    p.add_argument(
        "--ref_sinkhorn",
        action="store_true",
        help="Use GeomLoss Sinkhorn for --ref_attention_dir instead of cosine",
    )
    p.add_argument(
        "--ref_blur",
        type=float,
        default=0.05,
        help="GeomLoss blur for --ref_sinkhorn",
    )
    p.add_argument(
        "--ref_margin",
        type=float,
        default=0.25,
        help="Hard margin for Sinkhorn penalty (only penalize if distance > margin)",
    )
    p.add_argument(
        "--ref_layer_stride",
        type=int,
        default=1,
        help="Layer stride for --ref_sinkhorn",
    )
    p.add_argument(
        "--ref_step_mode",
        type=str,
        default="all",
        choices=["all", "first", "last"],
        help="Which steps to align for --ref_sinkhorn",
    )
    p.add_argument(
        "--ref_step_stride",
        type=int,
        default=1,
        help="Step stride for --ref_sinkhorn when step_mode=all",
    )
    p.add_argument(
        "--ref_type",
        type=str,
        default="ref",
        choices=["ref", "uniform", "random", "permute"],
        help="Reference type: 'ref' (loaded attention), 'uniform' (uniform distribution), "
        "'random' (random Gaussian normalized), 'permute' (permuted loaded attention)",
    )

    # Training parameters
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Use first N training examples (0=all)",
    )
    p.add_argument(
        "--max_valid_samples",
        type=int,
        default=0,
        help="Use first N validation examples (0=all)",
    )
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--max_source_length", type=int, default=256)
    p.add_argument("--max_target_length", type=int, default=256)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument(
        "--num_chunks_inside_batch",
        type=int,
        default=1,
        help="Split each batch into N chunks for forward/backward to reduce peak memory. "
        "Lambda is computed on full batch first, then chunks are processed with same lambda. "
        "Default=1 means no chunking (original behavior).",
    )
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--train_decoder_only",
        action="store_true",
        help="Only train decoder parameters, freeze encoder",
    )
    p.add_argument(
        "--train_from_scratch",
        action="store_true",
        help="Initialize model with random weights instead of loading pretrained weights",
    )
    p.add_argument(
        "--normalize_positions",
        action="store_true",
        help="Normalize position indices to [0,1] for position-based distance calculation",
    )
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Path to checkpoint directory to resume training from (for exact reproducibility)",
    )
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory (WARNING: disables attention gradients!)",
    )
    return p.parse_args()


def setup_logging(output_dir: str, log_to_file: bool = True):
    """Setup logging to both console and file."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"training_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")

    return logger


class MetricsLogger:
    """Logger for training metrics to CSV."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_log_file = os.path.join(output_dir, f"train_metrics_{timestamp}.csv")
        self.eval_log_file = os.path.join(output_dir, f"eval_metrics_{timestamp}.csv")

        # Initialize train log
        with open(self.train_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "epoch",
                    "loss",
                    "ce_loss",
                    "ref_penalty",
                    "before_margin",
                    "after_margin",
                    "effective_lambda",
                    "lr",
                ]
            )

        # Initialize eval log
        with open(self.eval_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "eval_loss"])

        logging.info(f"Metrics logging to:")
        logging.info(f"  Train: {self.train_log_file}")
        logging.info(f"  Eval:  {self.eval_log_file}")

    def log_train(
        self,
        step,
        epoch,
        loss,
        ce_loss,
        ref_penalty=None,
        before_margin=None,
        after_margin=None,
        effective_lambda=None,
        lr=None,
    ):
        with open(self.train_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    step,
                    epoch,
                    loss,
                    ce_loss,
                    ref_penalty if ref_penalty is not None else "",
                    before_margin if before_margin is not None else "",
                    after_margin if after_margin is not None else "",
                    effective_lambda if effective_lambda is not None else "",
                    lr if lr is not None else "",
                ]
            )

    def log_eval(self, step, epoch, eval_loss):
        with open(self.eval_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, epoch, eval_loss])


def save_training_state(
    checkpoint_dir: str,
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    args,
):
    """Save complete training state at epoch boundary for exact reproducibility."""
    import random
    import numpy as np

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Save training state (optimizer, scheduler, scaler, RNG states)
    training_state = {
        'epoch': epoch,
        'global_step': global_step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        # RNG states for exact reproducibility
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        # Save args for reference
        'args': vars(args),
    }

    state_path = os.path.join(checkpoint_dir, 'training_state.pt')
    torch.save(training_state, state_path)
    logging.info(f"Saved training state to {checkpoint_dir} (epoch={epoch}, global_step={global_step})")


def load_training_state(checkpoint_dir: str, optimizer, scheduler, scaler, device):
    """Load complete training state for exact reproducibility.

    Returns:
        tuple: (start_epoch, global_step) - training resumes from start_epoch
    """
    import random
    import numpy as np

    state_path = os.path.join(checkpoint_dir, 'training_state.pt')
    if not os.path.exists(state_path):
        logging.warning(f"No training state found at {state_path}")
        return 0, 0

    logging.info(f"Loading training state from {state_path}")
    # Load to CPU first to avoid issues with RNG states
    training_state = torch.load(state_path, map_location='cpu', weights_only=False)

    # Restore optimizer and scheduler states
    optimizer.load_state_dict(training_state['optimizer_state_dict'])
    scheduler.load_state_dict(training_state['scheduler_state_dict'])
    scaler.load_state_dict(training_state['scaler_state_dict'])

    # Restore RNG states for exact reproducibility
    torch.set_rng_state(training_state['rng_state'])
    if torch.cuda.is_available() and training_state.get('cuda_rng_state') is not None:
        torch.cuda.set_rng_state(training_state['cuda_rng_state'])
    if torch.cuda.is_available() and training_state.get('cuda_rng_state_all') is not None:
        torch.cuda.set_rng_state_all(training_state['cuda_rng_state_all'])

    # Restore python and numpy random states
    if 'python_rng_state' in training_state:
        random.setstate(training_state['python_rng_state'])
    if 'numpy_rng_state' in training_state:
        np.random.set_state(training_state['numpy_rng_state'])

    epoch = training_state['epoch']
    global_step = training_state['global_step']

    # Resume from next epoch
    start_epoch = epoch + 1
    logging.info(f"Resuming from epoch {start_epoch} (completed epoch {epoch}, global_step={global_step})")

    return start_epoch, global_step


def load_sibling_attention_references(file_path: str) -> Dict[str, List[torch.Tensor]]:
    """Load precomputed sibling attention references.

    Expected format: {lang: [[layer_tensors], ...]}
    Where each element is a list of layer attentions for one training sample.
    Each layer_tensor has shape [B, H, src_len].
    """
    if not file_path or not os.path.exists(file_path):
        # No warning - this is expected when using ref_attention_dir instead
        return {}

    logging.info(f"Loading sibling attention references from {file_path}")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Data should be a dict: {lang: [[layer_tensors], ...]}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid format: expected dict, got {type(data)}")

    logging.info(f"Loaded attention references for {len(data)} sibling languages")
    for lang, refs in data.items():
        num_samples = len(refs) if refs else 0
        num_layers = len(refs[0]) if refs and refs[0] else 0
        logging.info(f"  {lang}: {num_samples} samples, {num_layers} layers per sample")

    return data


def compute_sibling_penalty(
    cross_attentions: List[torch.Tensor],
    attention_mask_k: Optional[torch.Tensor],
    sibling_references: Dict[str, List[torch.Tensor]],
    penalty_type: str,
    **kwargs,
) -> torch.Tensor:
    """Compute sibling attention penalty using the specified method."""
    if not sibling_references:
        return torch.tensor(
            0.0, device=cross_attentions[0].device if cross_attentions else "cpu"
        )

    if penalty_type == "geomloss":
        return attention_sibling_penalty_geomloss_batched(
            cross_attentions,
            attention_mask_k,
            sibling_references,
            blur=kwargs.get("geomloss_blur", 0.05),
            layer_stride=kwargs.get("layer_stride", 1),
            normalize_positions=kwargs.get("normalize_positions", False),
        )
    else:
        raise ValueError(f"Unknown penalty type: {penalty_type}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    # Setup logging
    logger = setup_logging(args.output_dir, log_to_file=True)
    metrics_logger = MetricsLogger(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine model path: use checkpoint if resuming, otherwise use specified path
    resuming = args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint)
    if resuming:
        model_path = args.resume_from_checkpoint
        logging.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    else:
        model_path = args.model_name_or_path

    model, tokenizer = load_model_and_tokenizer(
        model_path,
        args.local_model_path,
        fp16=args.fp16,
        train_from_scratch=args.train_from_scratch if not resuming else False,
    )
    model.to(device)
    set_m2m_langs(model, tokenizer, args.src_lang, args.tgt_lang)

    # Enable gradient checkpointing to save memory (optional)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logging.info("Gradient checkpointing enabled (WARNING: attention gradients will be disabled!)")
    else:
        logging.info("Gradient checkpointing disabled (attention gradients enabled)")

    # Log training mode
    if args.train_from_scratch:
        logging.info("\n" + "=" * 80)
        logging.info("TRAINING MODE: Train from scratch (random initialization)")
        logging.info("=" * 80 + "\n")
    else:
        logging.info("\n" + "=" * 80)
        logging.info("TRAINING MODE: Fine-tuning (pretrained weights)")
        logging.info("=" * 80 + "\n")

    # Print model structure
    logging.info("\n" + "=" * 80)
    logging.info("MODEL STRUCTURE")
    logging.info("=" * 80)
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        logging.info(
            f"{name:60s} | Shape: {str(list(param.shape)):30s} | Params: {param_count:>12,}"
        )
    logging.info("-" * 80)
    logging.info(f"{'TOTAL PARAMETERS':60s} | {total_params:>12,}")
    logging.info("=" * 80 + "\n")

    # Hook to capture cross-attentions with gradients
    captured_cross_attentions = []

    def capture_cross_attention_hook(module, input, output):
        # output is typically (hidden_states, attention_weights, ...)
        if isinstance(output, tuple) and len(output) > 1:
            attn = output[1]  # attention weights
            if attn is not None and attn.requires_grad:
                captured_cross_attentions.append(attn)
        return output

    # Register hooks on decoder cross-attention layers
    if args.lambda_ref > 0:
        for name, module in model.named_modules():
            if "cross_attn" in name.lower() or "encoder_attn" in name.lower():
                module.register_forward_hook(capture_cross_attention_hook)
                if args.logging_steps > 0:
                    print(f"Registered hook on: {name}")

    # Load sibling attention references
    sibling_references = load_sibling_attention_references(args.sibling_attention_file)
    # External reference store
    ref_store = (
        AttentionRefStore(args.ref_attention_dir, args.ref_langs)
        if args.ref_attention_dir
        else None
    )

    # Setup datasets
    train_ds = ParallelTextDataset(args.train_file)
    valid_ds = ParallelTextDataset(args.valid_file)
    if args.max_train_samples and args.max_train_samples > 0:
        train_ds.examples = train_ds.examples[: args.max_train_samples]
    if args.max_valid_samples and args.max_valid_samples > 0:
        valid_ds.examples = valid_ds.examples[: args.max_valid_samples]
    collator = DataCollatorSeq2Seq(
        tokenizer, args.max_source_length, args.max_target_length
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # Freeze encoder if train_decoder_only is set
    if args.train_decoder_only:
        logging.info("\n" + "=" * 80)
        logging.info("FREEZING PARAMETERS - DECODER ONLY MODE")
        logging.info("=" * 80)
        # First freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Then unfreeze only decoder parameters
        for name, param in model.named_parameters():
            if "decoder" in name.lower():
                param.requires_grad = True

        # Log trainable vs frozen parameters with details
        logging.info("\nParameter Training Status:")
        logging.info("-" * 80)
        trainable_params = 0
        frozen_params = 0
        for name, param in model.named_parameters():
            param_count = param.numel()
            status = "✓ TRAINABLE" if param.requires_grad else "✗ FROZEN"
            if param.requires_grad:
                trainable_params += param_count
            else:
                frozen_params += param_count
            logging.info(f"{name:60s} | {status}")

        total_params = trainable_params + frozen_params
        logging.info("-" * 80)
        logging.info(
            f"Trainable parameters: {trainable_params:>12,} ({100 * trainable_params / total_params:.2f}%)"
        )
        logging.info(
            f"Frozen parameters:    {frozen_params:>12,} ({100 * frozen_params / total_params:.2f}%)"
        )
        logging.info(f"Total parameters:     {total_params:>12,}")
        logging.info("=" * 80 + "\n")
    else:
        logging.info("\n" + "=" * 80)
        logging.info("TRAINING MODE: All parameters (encoder + decoder)")
        logging.info("=" * 80 + "\n")

    # Setup optimizer and scheduler
    # Only optimize parameters that require gradients
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    t_total = (
        len(train_loader)
        * args.num_train_epochs
        // max(1, args.gradient_accumulation_steps)
    )
    warmup_steps = int(t_total * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, t_total)

    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.fp16 and torch.cuda.is_available()
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if resuming:
        start_epoch, global_step = load_training_state(
            args.resume_from_checkpoint, optimizer, scheduler, scaler, device
        )

    # Print blur recommendations based on configuration
    if args.lambda_ref > 0 and args.ref_sinkhorn:
        logging.info("\n" + "=" * 80)
        logging.info("POSITION AND BLUR CONFIGURATION")
        logging.info("=" * 80)
        logging.info(
            f"Position normalization: {'ENABLED' if args.normalize_positions else 'DISABLED'}"
        )
        logging.info(f"Current blur setting: {args.ref_blur}")

        # Calculate average source length from training data
        total_src_len = 0
        num_samples = min(100, len(train_ds.examples))  # Sample first 100 for speed
        for example in train_ds.examples[:num_samples]:
            src_tokens = tokenizer.encode(example[0], add_special_tokens=True)
            total_src_len += len(src_tokens)
        avg_src_len = total_src_len / num_samples if num_samples > 0 else 12.5

        logging.info(f"Average source length (sampled): {avg_src_len:.1f} tokens")

        if args.normalize_positions:
            # Normalized positions [0,1]
            recommended_blur = 0.1  # Typical good value for normalized positions
            blur_range = "0.1-0.5"
            logging.info(f"")
            logging.info(f"📊 NORMALIZED MODE:")
            logging.info(f"  Positions scaled to [0, 1]")
            logging.info(f"  Recommended blur range: {blur_range}")
            logging.info(f"  Suggested blur (TBD tuned): {recommended_blur}")
        else:
            # Non-normalized positions [0, S]
            # For average length S, good blur is typically 0.1-0.2 * S
            recommended_blur = avg_src_len * 0.15
            blur_min = avg_src_len * 0.1
            blur_max = avg_src_len * 0.5
            blur_range = f"{blur_min:.1f}-{blur_max:.1f}"

            logging.info(f"")
            logging.info(f"📊 NON-NORMALIZED MODE:")
            logging.info(f"  Positions use original indices [0, {avg_src_len:.0f}]")
            logging.info(f"  Recommended blur range: {blur_range}")
            logging.info(f"  Suggested blur (TBD tuned): {recommended_blur:.1f}")

    logging.info(f"Starting training with external reference attention penalty:")
    logging.info(f"  Reference penalty lambda: {args.lambda_ref}")
    logging.info(f"  Lambda mode: {args.lambda_ref_mode}")
    if args.lambda_ref_mode == "relative":
        logging.info(
            f"    → Lambda will be dynamically adjusted to maintain {args.lambda_ref:.1%} of CE loss"
        )
    else:
        logging.info(f"    → Lambda is fixed at {args.lambda_ref}")
    logging.info(f"  Reference languages: {args.ref_langs}")
    logging.info(f"  Reference directory: {args.ref_attention_dir}")
    if args.ref_sinkhorn:
        logging.info(
            f"  Penalty type: Sinkhorn (blur={args.ref_blur}, margin={args.ref_margin})"
        )
        logging.info(
            f"  Layer stride: {args.ref_layer_stride}, Step mode: {args.ref_step_mode}, Step stride: {args.ref_step_stride}"
        )
    else:
        logging.info(f"  Penalty type: Cosine")

    if args.num_chunks_inside_batch > 1:
        logging.info(
            f"  Memory optimization: batch split into {args.num_chunks_inside_batch} chunks"
        )
        logging.info(f"    → Lambda computed on full batch, then applied to each chunk")

    # Log ref_type configuration
    logging.info(f"  Reference type: {args.ref_type}")
    if args.ref_type == "uniform":
        logging.info("    → Using uniform distribution as reference (baseline)")
    elif args.ref_type == "random":
        logging.info("    → Using random Gaussian-sampled distribution as reference (baseline)")
    elif args.ref_type == "permute":
        logging.info("    → Using permuted loaded attention as reference (baseline)")
    else:
        logging.info("    → Using loaded reference attention")

    # Helper function to transform reference based on ref_type
    def get_effective_ref(
        ref_steps_loaded: List[torch.Tensor],
        ref_type: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        """Transform loaded reference attention based on ref_type.

        Args:
            ref_steps_loaded: Loaded reference attention (list of per-layer tensors)
            ref_type: Type of reference ('ref', 'uniform', 'random', 'permute')
            device: Device to create tensors on
            dtype: Data type for tensors

        Returns:
            Transformed reference attention

        Note:
            For 'uniform' and 'random', we use the loaded ref's length (not padded length).
            The refdir_sinkhorn_penalty_to_ref will handle crop/pad to align with current attention,
            then attn_mask_k will mask out padding positions before normalization.
        """
        if ref_type == "ref":
            # Use loaded reference as-is
            return ref_steps_loaded

        num_layers = len(ref_steps_loaded)
        # Get length from loaded ref (use first layer's length)
        ref_len = ref_steps_loaded[0].shape[-1] if ref_steps_loaded else 0

        if ref_type == "uniform":
            # Uniform distribution: equal weight for all positions
            uniform_attn = torch.ones(1, ref_len, device=device, dtype=dtype) / ref_len
            return [uniform_attn.clone() for _ in range(num_layers)]

        elif ref_type == "random":
            # Random Gaussian: sample from Gaussian, take absolute value, normalize
            ref_layers = []
            for _ in range(num_layers):
                random_attn = torch.abs(torch.randn(1, ref_len, device=device, dtype=dtype))
                random_attn = random_attn / (random_attn.sum(dim=-1, keepdim=True) + 1e-8)
                ref_layers.append(random_attn)
            return ref_layers

        elif ref_type == "permute":
            # Permute: shuffle the loaded attention values
            ref_layers = []
            for layer_attn in ref_steps_loaded:
                layer_attn = layer_attn.to(device=device, dtype=dtype)
                if layer_attn.dim() == 2:
                    # [1, S] -> permute along S dimension
                    perm_idx = torch.randperm(layer_attn.shape[-1], device=device)
                    permuted_attn = layer_attn[:, perm_idx]
                else:
                    permuted_attn = layer_attn.clone()
                ref_layers.append(permuted_attn)
            return ref_layers

        else:
            raise ValueError(f"Unknown ref_type: {ref_type}")

    # Helper function to compute loss for a batch/chunk
    def compute_chunk_loss(
        chunk_batch,
        ref_store,
        args,
        model,
        device,
        captured_cross_attentions,
        precomputed_lambda=None,
    ):
        """Compute CE loss and ref penalty for a chunk of the batch.

        Args:
            chunk_batch: The batch/chunk to process
            ref_store: Reference attention store
            args: Training arguments
            model: The model
            device: Device to use
            captured_cross_attentions: List to capture attentions via hook
            precomputed_lambda: If provided, use this lambda instead of computing

        Returns:
            total_loss, ce_loss, ext_ref_penalty, effective_lambda, ref_penalties_before, ref_penalties_after
        """
        captured_cross_attentions.clear()

        with torch.amp.autocast(
            "cuda", enabled=args.fp16 and torch.cuda.is_available()
        ):
            model_inputs = {k: v for k, v in chunk_batch.items() if k != "sample_idx"}
            outputs = model(
                **model_inputs,
                output_attentions=True,
                return_dict=True,
            )
            ce_loss = outputs.loss

        ce_loss = ce_loss.float()

        cross_atts = (
            outputs.cross_attentions
            if hasattr(outputs, "cross_attentions")
            and outputs.cross_attentions is not None
            else []
        )

        if cross_atts and not cross_atts[0].requires_grad and captured_cross_attentions:
            cross_atts = captured_cross_attentions

        total_loss = ce_loss
        ext_ref_penalty = None
        effective_lambda = 0.0
        ref_penalties_before = []
        ref_penalties_after = []

        if ref_store is not None and chunk_batch.get("sample_idx") is not None:
            penalty_context = (
                torch.no_grad() if args.lambda_ref == 0 else torch.enable_grad()
            )

            with penalty_context:
                ref_penalties = []
                sample_idx = chunk_batch["sample_idx"].tolist()
                for b_i, idx_val in enumerate(sample_idx):
                    curr_per_layer = [la[b_i : b_i + 1] for la in cross_atts]
                    ref_steps_any = None
                    for lang in args.ref_langs or []:
                        ref_steps = ref_store.load_by_index(idx_val, lang)
                        if ref_steps is not None and len(ref_steps) > 0:
                            ref_steps_any = ref_steps
                            break
                    if ref_steps_any is None or len(ref_steps_any) == 0:
                        continue
                    # Transform reference based on ref_type
                    ref_steps_effective = get_effective_ref(
                        ref_steps_any,
                        args.ref_type,
                        device,
                        curr_per_layer[0].dtype if curr_per_layer else torch.float32,
                    )
                    am_q = chunk_batch.get("decoder_attention_mask")
                    if am_q is None and chunk_batch.get("labels") is not None:
                        am_q = chunk_batch["labels"][b_i : b_i + 1].ne(-100)
                    else:
                        am_q = am_q[b_i : b_i + 1] if am_q is not None else None
                    am_k = chunk_batch.get("attention_mask")
                    am_k = am_k[b_i : b_i + 1] if am_k is not None else None
                    if args.ref_sinkhorn:
                        (
                            ref_pen,
                            ref_pen_before,
                            ref_pen_after,
                        ) = refdir_sinkhorn_penalty_to_ref(
                            curr_per_layer,
                            ref_steps_effective,
                            attn_mask_q=am_q,
                            attn_mask_k=am_k,
                            blur=args.ref_blur,
                            margin=args.ref_margin,
                            layer_stride=args.ref_layer_stride,
                            step_mode=args.ref_step_mode,
                            step_stride=args.ref_step_stride,
                            normalize_positions=args.normalize_positions,
                        )
                        ref_penalties_before.append(ref_pen_before)
                        ref_penalties_after.append(ref_pen_after)
                    else:
                        ref_pen = cosine_penalty_to_ref(
                            curr_per_layer,
                            ref_steps_effective,
                            attn_mask_q=am_q,
                            attn_mask_k=am_k,
                            step_mode="all",
                        )
                    ref_penalties.append(ref_pen)

                if ref_penalties:
                    ext_ref_penalty = torch.stack(ref_penalties).mean()

                    if args.lambda_ref > 0:
                        if precomputed_lambda is not None:
                            # Use precomputed lambda from full batch
                            effective_lambda = precomputed_lambda
                        elif args.lambda_ref_mode == "relative":
                            effective_lambda = args.lambda_ref * (
                                ce_loss.detach() / (ext_ref_penalty.detach() + 1e-8)
                            )
                        else:
                            effective_lambda = args.lambda_ref

                        total_loss = total_loss + effective_lambda * ext_ref_penalty
                    else:
                        effective_lambda = 0.0

        return (
            total_loss,
            ce_loss,
            ext_ref_penalty,
            effective_lambda,
            ref_penalties_before,
            ref_penalties_after,
        )

    # Helper function to split batch into chunks
    def split_batch(batch, num_chunks):
        """Split a batch into num_chunks smaller batches."""
        batch_size = batch["input_ids"].shape[0]
        chunk_size = (batch_size + num_chunks - 1) // num_chunks
        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, batch_size)
            if start_idx >= batch_size:
                break
            chunk = {k: v[start_idx:end_idx] for k, v in batch.items()}
            chunks.append(chunk)
        return chunks

    # global_step is set above (0 or resumed value)
    model.train()
    for epoch in range(start_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = batch["input_ids"].shape[0]

            # Determine if we need chunked processing
            use_chunking = args.num_chunks_inside_batch > 1 and batch_size > 1

            if use_chunking:
                # === Chunked backward for memory efficiency ===
                # Forward once, backward in chunks with retain_graph

                # Step 1: Forward full batch once
                captured_cross_attentions.clear()

                with torch.amp.autocast(
                    "cuda", enabled=args.fp16 and torch.cuda.is_available()
                ):
                    model_inputs = {k: v for k, v in batch.items() if k != "sample_idx"}
                    outputs = model(
                        **model_inputs,
                        output_attentions=True,
                        return_dict=True,
                    )

                # Get logits for manual per-chunk loss computation
                logits = outputs.logits  # [B, T, V]
                labels = batch["labels"]  # [B, T]

                cross_atts = (
                    outputs.cross_attentions
                    if hasattr(outputs, "cross_attentions")
                    and outputs.cross_attentions is not None
                    else []
                )

                if (
                    cross_atts
                    and not cross_atts[0].requires_grad
                    and captured_cross_attentions
                ):
                    cross_atts = captured_cross_attentions

                # Step 2: Get CE loss
                ce_loss = outputs.loss.float()

                # Debug: print memory after forward
                if step < 3:
                    mem_alloc = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    print(
                        f"[Step {step}] After forward: allocated={mem_alloc:.2f}GB, reserved={mem_reserved:.2f}GB"
                    )
                    print(f"  ce_loss = {ce_loss.item():.4f}")

                # First pass: compute geo loss with no_grad to get effective_lambda
                ref_penalties_before = []
                ref_penalties_after = []
                effective_lambda = 0.0
                ext_ref_penalty = None

                num_chunks = args.num_chunks_inside_batch
                chunk_size = batch_size // num_chunks

                if (
                    ref_store is not None
                    and batch.get("sample_idx") is not None
                    and args.lambda_ref > 0
                ):
                    # Compute full batch geo loss with no_grad to get effective_lambda
                    with torch.no_grad():
                        sample_idx = batch["sample_idx"].tolist()
                        all_ref_penalties = []
                        for b_i, idx_val in enumerate(sample_idx):
                            curr_per_layer = [la[b_i : b_i + 1] for la in cross_atts]
                            ref_steps_any = None
                            for lang in args.ref_langs or []:
                                ref_steps = ref_store.load_by_index(idx_val, lang)
                                if ref_steps is not None and len(ref_steps) > 0:
                                    ref_steps_any = ref_steps
                                    break
                            if ref_steps_any is None or len(ref_steps_any) == 0:
                                all_ref_penalties.append(None)
                                continue
                            # Transform reference based on ref_type
                            ref_steps_effective = get_effective_ref(
                                ref_steps_any,
                                args.ref_type,
                                device,
                                curr_per_layer[0].dtype if curr_per_layer else torch.float32,
                            )
                            am_q = batch.get("decoder_attention_mask")
                            if am_q is None and batch.get("labels") is not None:
                                am_q = batch["labels"][b_i : b_i + 1].ne(-100)
                            else:
                                am_q = am_q[b_i : b_i + 1] if am_q is not None else None
                            am_k = batch.get("attention_mask")
                            am_k = am_k[b_i : b_i + 1] if am_k is not None else None
                            if args.ref_sinkhorn:
                                (
                                    ref_pen,
                                    ref_pen_before,
                                    ref_pen_after,
                                ) = refdir_sinkhorn_penalty_to_ref(
                                    curr_per_layer,
                                    ref_steps_effective,
                                    attn_mask_q=am_q,
                                    attn_mask_k=am_k,
                                    blur=args.ref_blur,
                                    margin=args.ref_margin,
                                    layer_stride=args.ref_layer_stride,
                                    step_mode=args.ref_step_mode,
                                    step_stride=args.ref_step_stride,
                                    normalize_positions=args.normalize_positions,
                                )
                                ref_penalties_before.append(ref_pen_before)
                                ref_penalties_after.append(ref_pen_after)
                            else:
                                ref_pen = cosine_penalty_to_ref(
                                    curr_per_layer,
                                    ref_steps_effective,
                                    attn_mask_q=am_q,
                                    attn_mask_k=am_k,
                                    step_mode="all",
                                )
                            all_ref_penalties.append(
                                ref_pen.item() if ref_pen is not None else None
                            )

                        # Compute effective_lambda from full batch
                        valid_penalties_vals = [
                            p for p in all_ref_penalties if p is not None
                        ]
                        if valid_penalties_vals:
                            ext_ref_penalty_val = sum(valid_penalties_vals) / len(
                                valid_penalties_vals
                            )
                            if args.lambda_ref_mode == "relative":
                                effective_lambda = args.lambda_ref * (
                                    ce_loss.item() / (ext_ref_penalty_val + 1e-8)
                                )
                            else:
                                effective_lambda = args.lambda_ref
                            # Store for logging
                            ext_ref_penalty = torch.tensor(
                                ext_ref_penalty_val, device=device
                            )

                    if step < 3:
                        mem_alloc = torch.cuda.memory_allocated() / 1e9
                        mem_reserved = torch.cuda.memory_reserved() / 1e9
                        print(
                            f"[Step {step}] After no_grad geo_loss computation: allocated={mem_alloc:.2f}GB, reserved={mem_reserved:.2f}GB"
                        )
                        if ext_ref_penalty is not None:
                            print(f"  ext_ref_penalty = {ext_ref_penalty.item():.4f}")
                            print(f"  effective_lambda = {effective_lambda:.4f}")

                # CE loss backward once, geo loss in chunks
                has_geo_loss = (
                    ref_store is not None
                    and batch.get("sample_idx") is not None
                    and args.lambda_ref > 0
                    and effective_lambda > 0
                )

                # CE loss backward (once, keep graph for geo loss if needed)
                ce_loss_scaled = ce_loss / args.gradient_accumulation_steps
                if has_geo_loss:
                    scaler.scale(ce_loss_scaled).backward(retain_graph=True)
                    if step < 3:
                        mem_alloc = torch.cuda.memory_allocated() / 1e9
                        mem_reserved = torch.cuda.memory_reserved() / 1e9
                        print(
                            f"[Step {step}] After CE backward (retain_graph=True): "
                            f"allocated={mem_alloc:.2f}GB, reserved={mem_reserved:.2f}GB"
                        )
                else:
                    scaler.scale(ce_loss_scaled).backward()
                    if step < 3:
                        mem_alloc = torch.cuda.memory_allocated() / 1e9
                        mem_reserved = torch.cuda.memory_reserved() / 1e9
                        print(
                            f"[Step {step}] After CE backward: "
                            f"allocated={mem_alloc:.2f}GB, reserved={mem_reserved:.2f}GB"
                        )

                # Chunked geo loss: compute and backward one chunk at a time
                if has_geo_loss:
                    sample_idx = batch["sample_idx"].tolist()

                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = (chunk_idx + 1) * chunk_size
                        is_last_chunk = chunk_idx == num_chunks - 1

                        # Compute geo loss for this chunk only (with grad)
                        chunk_ref_penalties = []
                        for b_i in range(start_idx, end_idx):
                            idx_val = sample_idx[b_i]
                            curr_per_layer = [la[b_i : b_i + 1] for la in cross_atts]
                            ref_steps_any = None
                            for lang in args.ref_langs or []:
                                ref_steps = ref_store.load_by_index(idx_val, lang)
                                if ref_steps is not None and len(ref_steps) > 0:
                                    ref_steps_any = ref_steps
                                    break
                            if ref_steps_any is None or len(ref_steps_any) == 0:
                                continue
                            # Transform reference based on ref_type
                            ref_steps_effective = get_effective_ref(
                                ref_steps_any,
                                args.ref_type,
                                device,
                                curr_per_layer[0].dtype if curr_per_layer else torch.float32,
                            )
                            am_q = batch.get("decoder_attention_mask")
                            if am_q is None and batch.get("labels") is not None:
                                am_q = batch["labels"][b_i : b_i + 1].ne(-100)
                            else:
                                am_q = am_q[b_i : b_i + 1] if am_q is not None else None
                            am_k = batch.get("attention_mask")
                            am_k = am_k[b_i : b_i + 1] if am_k is not None else None
                            if args.ref_sinkhorn:
                                ref_pen, _, _ = refdir_sinkhorn_penalty_to_ref(
                                    curr_per_layer,
                                    ref_steps_effective,
                                    attn_mask_q=am_q,
                                    attn_mask_k=am_k,
                                    blur=args.ref_blur,
                                    margin=args.ref_margin,
                                    layer_stride=args.ref_layer_stride,
                                    step_mode=args.ref_step_mode,
                                    step_stride=args.ref_step_stride,
                                    normalize_positions=args.normalize_positions,
                                )
                            else:
                                ref_pen = cosine_penalty_to_ref(
                                    curr_per_layer,
                                    ref_steps_effective,
                                    attn_mask_q=am_q,
                                    attn_mask_k=am_k,
                                    step_mode="all",
                                )
                            chunk_ref_penalties.append(ref_pen)

                        if not chunk_ref_penalties:
                            continue

                        chunk_ref_penalty = torch.stack(chunk_ref_penalties).mean()
                        chunk_geo_loss = (
                            effective_lambda
                            * chunk_ref_penalty
                            / (num_chunks * args.gradient_accumulation_steps)
                        )

                        # NaN/Inf detection
                        if torch.isnan(chunk_geo_loss) or torch.isinf(chunk_geo_loss):
                            logging.warning(
                                f"NaN/Inf in geo_loss at step {global_step} chunk {chunk_idx}, skipping."
                            )
                            continue

                        # Backward: retain_graph for all but last chunk
                        retain = not is_last_chunk
                        scaler.scale(chunk_geo_loss).backward(retain_graph=retain)

                        if step < 3:
                            mem_alloc = torch.cuda.memory_allocated() / 1e9
                            mem_reserved = torch.cuda.memory_reserved() / 1e9
                            print(
                                f"[Step {step}] After geo chunk {chunk_idx} backward (retain={retain}): "
                                f"allocated={mem_alloc:.2f}GB, reserved={mem_reserved:.2f}GB"
                            )
                            print(f"  chunk_geo_loss = {chunk_geo_loss.item():.4f}")

                # For logging
                total_loss = ce_loss
                if ext_ref_penalty is not None and args.lambda_ref > 0:
                    total_loss = total_loss + effective_lambda * ext_ref_penalty
                total_loss = total_loss / args.gradient_accumulation_steps

            else:
                # === Original non-chunked processing ===
                captured_cross_attentions.clear()

                with torch.amp.autocast(
                    "cuda", enabled=args.fp16 and torch.cuda.is_available()
                ):
                    model_inputs = {k: v for k, v in batch.items() if k != "sample_idx"}
                    outputs = model(
                        **model_inputs,
                        output_attentions=True,
                        return_dict=True,
                    )
                    ce_loss = outputs.loss

                ce_loss = ce_loss.float()

                cross_atts = (
                    outputs.cross_attentions
                    if hasattr(outputs, "cross_attentions")
                    and outputs.cross_attentions is not None
                    else []
                )

                use_captured = False
                if (
                    cross_atts
                    and not cross_atts[0].requires_grad
                    and captured_cross_attentions
                ):
                    cross_atts = captured_cross_attentions
                    use_captured = True

                if global_step == 0 and cross_atts:
                    print(f"\n{'='*70}")
                    print(f"[GRADIENT CHECK - Step {global_step}]")
                    print(f"  ce_loss.requires_grad: {ce_loss.requires_grad}")
                    print(
                        f"  cross_atts[0].requires_grad: {cross_atts[0].requires_grad}"
                    )
                    print(f"  cross_atts[0].is_leaf: {cross_atts[0].is_leaf}")
                    print(f"  cross_atts[0].device: {cross_atts[0].device}")
                    print(f"  cross_atts[0].dtype: {cross_atts[0].dtype}")
                    print(f"  Using captured attentions: {use_captured}")
                    if not cross_atts[0].requires_grad:
                        print(f"  ⚠️  WARNING: cross_atts STILL has no gradient!")
                        print(f"  ⚠️  The attention penalty will NOT affect training!")
                    else:
                        print(f"  ✅ cross_atts has gradient - penalty will work!")
                    print(f"{'='*70}\n")

                total_loss = ce_loss
                ext_ref_penalty = None
                effective_lambda = 0.0
                ref_penalties_before = []
                ref_penalties_after = []

                if ref_store is not None and batch.get("sample_idx") is not None:
                    penalty_context = (
                        torch.no_grad() if args.lambda_ref == 0 else torch.enable_grad()
                    )

                    with penalty_context:
                        ref_penalties = []
                        sample_idx = batch["sample_idx"].tolist()
                        for b_i, idx_val in enumerate(sample_idx):
                            curr_per_layer = [la[b_i : b_i + 1] for la in cross_atts]
                            ref_steps_any = None
                            for lang in args.ref_langs or []:
                                ref_steps = ref_store.load_by_index(idx_val, lang)
                                if ref_steps is not None and len(ref_steps) > 0:
                                    ref_steps_any = ref_steps
                                    break
                            if ref_steps_any is None or len(ref_steps_any) == 0:
                                continue
                            # Transform reference based on ref_type
                            ref_steps_effective = get_effective_ref(
                                ref_steps_any,
                                args.ref_type,
                                device,
                                curr_per_layer[0].dtype if curr_per_layer else torch.float32,
                            )
                            am_q = batch.get("decoder_attention_mask")
                            if am_q is None and batch.get("labels") is not None:
                                am_q = batch["labels"][b_i : b_i + 1].ne(-100)
                            else:
                                am_q = am_q[b_i : b_i + 1] if am_q is not None else None
                            am_k = batch.get("attention_mask")
                            am_k = am_k[b_i : b_i + 1] if am_k is not None else None
                            if args.ref_sinkhorn:
                                (
                                    ref_pen,
                                    ref_pen_before,
                                    ref_pen_after,
                                ) = refdir_sinkhorn_penalty_to_ref(
                                    curr_per_layer,
                                    ref_steps_effective,
                                    attn_mask_q=am_q,
                                    attn_mask_k=am_k,
                                    blur=args.ref_blur,
                                    margin=args.ref_margin,
                                    layer_stride=args.ref_layer_stride,
                                    step_mode=args.ref_step_mode,
                                    step_stride=args.ref_step_stride,
                                    normalize_positions=args.normalize_positions,
                                )
                                ref_penalties_before.append(ref_pen_before)
                                ref_penalties_after.append(ref_pen_after)
                            else:
                                ref_pen = cosine_penalty_to_ref(
                                    curr_per_layer,
                                    ref_steps_effective,
                                    attn_mask_q=am_q,
                                    attn_mask_k=am_k,
                                    step_mode="all",
                                )
                            ref_penalties.append(ref_pen)
                        if ref_penalties:
                            ext_ref_penalty = torch.stack(ref_penalties).mean()

                            if args.lambda_ref > 0:
                                if args.lambda_ref_mode == "relative":
                                    effective_lambda = args.lambda_ref * (
                                        ce_loss.detach()
                                        / (ext_ref_penalty.detach() + 1e-8)
                                    )
                                else:
                                    effective_lambda = args.lambda_ref

                                total_loss = (
                                    total_loss + effective_lambda * ext_ref_penalty
                                )
                            else:
                                effective_lambda = 0.0

                            if global_step == 0:
                                print(f"\n{'='*70}")
                                print(f"[PENALTY GRADIENT CHECK - Step {global_step}]")
                                print(f"  ce_loss: {ce_loss.item():.6f}")
                                print(
                                    f"  ext_ref_penalty.requires_grad: {ext_ref_penalty.requires_grad}"
                                )
                                print(
                                    f"  ext_ref_penalty.item(): {ext_ref_penalty.item():.6f}"
                                )
                                print(
                                    f"  total_loss.requires_grad: {total_loss.requires_grad}"
                                )
                                print(f"  lambda_ref (base): {args.lambda_ref}")
                                print(f"  lambda_ref_mode: {args.lambda_ref_mode}")
                                if args.lambda_ref > 0:
                                    if args.lambda_ref_mode == "relative":
                                        print(
                                            f"  effective_lambda (dynamic): {effective_lambda:.6f}"
                                        )
                                        print(
                                            f"  penalty contribution: {(effective_lambda * ext_ref_penalty).item():.6f}"
                                        )
                                        print(
                                            f"  ratio (penalty/CE): {(effective_lambda * ext_ref_penalty / ce_loss).item():.2%}"
                                        )
                                    else:
                                        print(
                                            f"  effective_lambda (fixed): {effective_lambda:.6f}"
                                        )
                                        print(
                                            f"  penalty contribution: {(effective_lambda * ext_ref_penalty).item():.6f}"
                                        )
                                else:
                                    print(
                                        f"  ⚠️  lambda_ref=0, penalty NOT added to loss (logging only)"
                                    )
                                print(f"{'='*70}\n")

                total_loss = total_loss / args.gradient_accumulation_steps

            # NaN/Inf detection to prevent parameter corruption
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                ref_penalty_info = ""
                if ref_store is not None and ext_ref_penalty is not None:
                    ref_penalty_info = f", ref_penalty={ext_ref_penalty.item()}"
                logging.warning(
                    f"NaN/Inf detected at step {global_step}, skipping backward. "
                    f"total_loss={total_loss.item()}, ce_loss={ce_loss.item()}{ref_penalty_info}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            # Backward pass (skip if already done in chunked processing)
            if not use_chunking:
                scaler.scale(total_loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Prepare metrics
                    current_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
                    total_loss_val = (
                        total_loss.item() * args.gradient_accumulation_steps
                    )
                    ce_loss_val = ce_loss.item()
                    ref_penalty_val = None
                    avg_before = None
                    avg_after = None
                    effective_lambda_val = None

                    log_msg = f"step {global_step} | loss {total_loss_val:.4f} | ce {ce_loss_val:.4f}"

                    # Log ref_penalty even when lambda_ref=0 (for comparison/debugging)
                    if ref_store is not None and ext_ref_penalty is not None:
                        ref_penalty_val = ext_ref_penalty.item()
                        log_msg += f" | ref_penalty {ref_penalty_val:.4f}"

                        # Add effective lambda info
                        if effective_lambda is not None:
                            effective_lambda_val = (
                                effective_lambda
                                if isinstance(effective_lambda, (float, int))
                                else effective_lambda.item()
                            )
                            if args.lambda_ref_mode == "relative":
                                log_msg += f" | λ_eff {effective_lambda_val:.4f}"
                            else:
                                log_msg += f" | λ {effective_lambda_val:.4f}"

                        # Add lambda info if it's 0 (for clarity)
                        if args.lambda_ref == 0:
                            log_msg += " (not applied)"

                        # Add before/after margin info if using Sinkhorn
                        if args.ref_sinkhorn and ref_penalties_before:
                            avg_before = sum(ref_penalties_before) / len(
                                ref_penalties_before
                            )
                            avg_after = sum(ref_penalties_after) / len(
                                ref_penalties_after
                            )
                            log_msg += f" | before_margin {avg_before:.4f} | after_margin {avg_after:.4f}"

                    # Log to console and file
                    logging.info(log_msg)

                    # Log to CSV
                    metrics_logger.log_train(
                        step=global_step,
                        epoch=epoch,
                        loss=total_loss_val,
                        ce_loss=ce_loss_val,
                        ref_penalty=ref_penalty_val,
                        before_margin=avg_before,
                        after_margin=avg_after,
                        effective_lambda=effective_lambda_val,
                        lr=current_lr,
                    )

            if (
                args.eval_steps > 0
                and global_step % args.eval_steps == 0
                and global_step > 0
            ):
                model.eval()
                eval_loss = 0.0
                eval_steps = 0
                with torch.no_grad():
                    for vb in valid_loader:
                        vb = {k: v.to(device) for k, v in vb.items()}
                        eval_inputs = {k: v for k, v in vb.items() if k != "sample_idx"}
                        out = model(**eval_inputs, output_attentions=False)
                        eval_loss += out.loss.item()
                        eval_steps += 1
                eval_loss /= max(1, eval_steps)
                logging.info(f"eval | loss {eval_loss:.4f}")
                metrics_logger.log_eval(
                    step=global_step, epoch=epoch, eval_loss=eval_loss
                )
                model.train()

            if (
                args.save_steps > 0
                and global_step % args.save_steps == 0
                and global_step > 0
            ):
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

        # Save checkpoint at end of each epoch (for exact reproducibility resume)
        epoch_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
        save_training_state(
            epoch_checkpoint_dir,
            model,
            tokenizer,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            args,
        )

    # Final save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info(f"Training completed. Model saved to {args.output_dir}")
    logging.info(f"Logs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
