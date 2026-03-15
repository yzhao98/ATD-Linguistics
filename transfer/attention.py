from typing import List, Optional, Dict
import os
import pickle

import torch
import torch.nn.functional as F


def attention_sibling_penalty_geomloss_batched(
    cross_attentions: List[torch.Tensor],
    attention_mask_k: Optional[torch.Tensor],
    sibling_references: Dict[str, List[torch.Tensor]],
    blur: float = 0.05,
    layer_stride: int = 1,
    normalize_positions: bool = False,
) -> torch.Tensor:
    """
    Batched GeomLoss penalty with proper handling of variable-length sequences.

    For each layer:
    1. Take mean over target dimension: [B, H, tgt_len, src_len] -> [B, H, src_len]
    2. Mean over heads: [B, H, src_len] -> [B, src_len] (for speed)
    3. Normalize attention weights to sum to 1 (accounting for padding via attention_mask_k)
    4. Compute Wasserstein-2 distance between current and reference attention using GeomLoss

    Strategy for parallelization:
        - Batch samples may have different lengths → cannot batch across samples
        - Same sample's different layers have same length → CAN batch across layers
        - For each sample, collect all layers and compute in one batched call

    Args:
        cross_attentions: List of cross-attention tensors [B, H, tgt_len, src_len]
        attention_mask_k: Source attention mask [B, src_len] (1 for real tokens, 0 for padding)
        sibling_references: Dict of precomputed reference attentions {lang: [layer tensors]}
        blur: GeomLoss blur parameter (related to kernel bandwidth, ~0.05 for normalized distances)
        layer_stride: Process every k-th layer to speed up

    Returns:
        Average Wasserstein-2 distance across all layers and references

    Note:
        - Heads are always averaged before computing distance
        - Each sample in batch may have different src_len (handled by attention_mask_k)
        - For same sample, all layers are batched together for efficient computation
    """
    try:
        from geomloss import SamplesLoss
    except Exception as e:
        raise ImportError("geomloss is required: pip install geomloss")

    if len(cross_attentions) == 0 or len(sibling_references) == 0:
        return torch.tensor(
            0.0, device=cross_attentions[0].device if cross_attentions else "cpu"
        )

    device = cross_attentions[0].device
    dtype = cross_attentions[0].dtype
    eps = 1e-8

    # GeomLoss Sinkhorn loss function
    # p=2 for Wasserstein-2, blur is the smoothing scale
    # debias=False to match standard Sinkhorn (debias=True adds extra correction term)
    sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=blur, debias=False)

    layer_indices = list(range(0, len(cross_attentions), max(1, layer_stride)))
    num_layers_to_process = len(layer_indices)

    # First pass: collect and preprocess all layers
    # Structure: {(batch_idx, ref_lang): {'current': [L, actual_len], 'ref': [L, actual_len], 'len': actual_len}}
    sample_data = {}

    for layer_idx in layer_indices:
        current_attn = cross_attentions[layer_idx]  # [B, H, tgt_len, src_len]

        # Average over target dimension and heads: [B, H, tgt_len, src_len] -> [B, src_len]
        current_avg = current_attn.mean(dim=2).mean(dim=1)  # [B, src_len]
        B, src_len = current_avg.shape

        # Process each reference language
        for ref_lang, ref_layers in sibling_references.items():
            if layer_idx >= len(ref_layers):
                continue

            ref_attn = ref_layers[layer_idx]  # [B, H, src_len] or [B, src_len]

            # Average heads in reference if needed
            if ref_attn.dim() == 3:
                ref_attn = ref_attn.mean(dim=1)  # [B, src_len]

            # Apply attention mask and normalize
            if attention_mask_k is not None:
                mask = attention_mask_k  # [B, src_len]
                current_masked = current_avg * mask
                ref_masked = ref_attn * mask
            else:
                current_masked = current_avg
                ref_masked = ref_attn

            # Normalize to probability distributions
            current_norm = current_masked / (
                current_masked.sum(dim=-1, keepdim=True) + eps
            )
            ref_norm = ref_masked / (ref_masked.sum(dim=-1, keepdim=True) + eps)

            # Collect data for each batch sample
            for b in range(B):
                # Get actual length for this sample
                if attention_mask_k is not None:
                    actual_len = int(attention_mask_k[b].sum().item())
                    if actual_len == 0:
                        continue
                else:
                    actual_len = src_len

                key = (b, ref_lang, actual_len)
                if key not in sample_data:
                    sample_data[key] = {
                        "current": [],
                        "ref": [],
                    }

                # Collect attention weights for valid positions
                sample_data[key]["current"].append(
                    current_norm[b, :actual_len]
                )  # [actual_len]
                sample_data[key]["ref"].append(ref_norm[b, :actual_len])  # [actual_len]

    # Second pass: batch compute distances for each sample
    all_distances = []

    for (b, ref_lang, actual_len), data in sample_data.items():
        num_collected_layers = len(data["current"])
        if num_collected_layers == 0:
            continue

        # Stack all layers for this sample: [num_layers, actual_len]
        curr_weights_batch = torch.stack(data["current"], dim=0)  # [L, actual_len]
        ref_weights_batch = torch.stack(data["ref"], dim=0)  # [L, actual_len]

        # Positions: normalized [0,1] or original indices [0,actual_len-1]
        if normalize_positions:
            # Normalized positions: [0, 1]
            positions = torch.linspace(
                0, 1, actual_len, device=device, dtype=dtype
            ).view(actual_len, 1)
        else:
            # Original indices: [0, actual_len-1]
            positions = torch.arange(actual_len, device=device, dtype=dtype).view(
                actual_len, 1
            )

        # Expand positions for batch: [1, actual_len, 1] -> [L, actual_len, 1]
        positions_batch = positions.unsqueeze(0).expand(num_collected_layers, -1, -1)

        # Reshape for GeomLoss batch mode: [L, actual_len, 1] and [L, actual_len]
        curr_weights_batch = curr_weights_batch.unsqueeze(-1)  # [L, actual_len, 1]
        ref_weights_batch = ref_weights_batch.unsqueeze(-1)  # [L, actual_len, 1]

        # Batched Sinkhorn computation: returns [L] tensor
        # GeomLoss API: (weights1, positions1, weights2, positions2)
        w2_squared_batch = sinkhorn_loss(
            curr_weights_batch, positions_batch, ref_weights_batch, positions_batch
        )
        w2_dist_batch = torch.sqrt(w2_squared_batch + eps)  # [L]

        # Add all distances for this sample
        all_distances.extend(w2_dist_batch.unbind())

    if len(all_distances) == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Return average distance across all samples, layers, and references
    return torch.stack(all_distances).mean()


class AttentionRefStore:
    """Load per-sentence cross_attentions or reduced attention.
    Directory layout: <dir>/<idx>.pkl.
    Priority: attention_reduced[lang]['per_layer_avg'] -> attention_data[lang]['cross_attentions']
    """

    def __init__(self, root_dir: str, ref_langs: Optional[List[str]] = None) -> None:
        self.root_dir = root_dir
        self.ref_langs = set(ref_langs) if ref_langs else None

    def load_by_index(self, idx: int, lang: str) -> Optional[List]:
        path = os.path.join(self.root_dir, f"{idx}.pkl")
        if not os.path.exists(path):
            print(f"[DEBUG] File not found: {path}")
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            # Handle corrupted pickle files gracefully
            print(f"[WARNING] Failed to load sample idx={idx}, path={path}: {e}")
            return None
        # Read ["attention_reduced"][lang]["per_layer_avg"]
        try:
            if "attention_reduced" in data:
                if lang in data["attention_reduced"]:
                    lang_data = data["attention_reduced"][lang]
                    if "per_layer_avg" in lang_data:
                        result = lang_data["per_layer_avg"]
                        # Convert single tensor [L,S] to List[Tensor] of [1,S]
                        if isinstance(result, torch.Tensor):
                            if result.dim() == 2:  # [L,S]
                                result_list = [
                                    result[i : i + 1, :] for i in range(result.shape[0])
                                ]
                                return result_list
                            else:
                                return None
                        elif isinstance(result, list):
                            first_shape = (
                                result[0].shape
                                if result and isinstance(result[0], torch.Tensor)
                                else "?"
                            )
                            return result
                        else:
                            print(f"[DEBUG] Unexpected type: {type(result)}")
                            return None
                    else:
                        print(f"[DEBUG] 'per_layer_avg' not found in lang '{lang}'")
                else:
                    print(f"[DEBUG] Lang '{lang}' not found in attention_reduced")
            else:
                print(f"[DEBUG] 'attention_reduced' not found in data")
            return None
        except (KeyError, TypeError) as e:
            print(f"[DEBUG] Exception: {e}")
            return None


def cosine_penalty_to_ref(
    current_cross: List[torch.Tensor],  # per-layer [B,H,T,S]
    ref_steps: List[List[torch.Tensor]],  # steps x layers, each [1,H,1,S_ref]
    attn_mask_q: Optional[torch.Tensor],  # [B,T]
    attn_mask_k: Optional[torch.Tensor],  # [B,S]
    step_mode: str = "all",
) -> torch.Tensor:
    """Compute cosine penalty between current attention and a reference.
    - current_cross: model forward cross_attentions per layer
    - ref_steps: reference steps (choose one step, e.g., first)
    - Align src length by cropping/padding reference along key dimension
    """
    if len(current_cross) == 0 or len(ref_steps) == 0:
        return torch.tensor(
            0.0, device=current_cross[0].device if current_cross else "cpu"
        )

    B = current_cross[0].shape[0]
    device = current_cross[0].device
    eps = 1e-8
    penalties = []
    # step selection
    step_indices = (
        range(len(ref_steps))
        if step_mode == "all"
        else ([0] if step_mode == "first" else [-1])
    )
    for step_idx in step_indices:
        ref_layers = ref_steps[step_idx]
        for layer_idx, curr in enumerate(current_cross):
            # curr: [B,H,T,S]
            H = curr.shape[1]
            T = curr.shape[-2]
            S = curr.shape[-1]
            ref = ref_layers[layer_idx] if layer_idx < len(ref_layers) else None
            if ref is None:
                continue
            # ref: [1,H,1,S_ref]
            ref = ref.to(device)
            S_ref = ref.shape[-1]
            ref_exp = ref.repeat(B, 1, T, 1)  # [B,H,T,S_ref]
            if S_ref > S:
                ref_exp = ref_exp[..., :S]
            elif S_ref < S:
                pad = (0, S - S_ref)
                ref_exp = F.pad(ref_exp, pad)
            attn_curr = curr
            attn_ref = ref_exp
            if attn_mask_q is not None:
                qm = attn_mask_q[:, None, :, None]
                attn_curr = attn_curr * qm
                attn_ref = attn_ref * qm
            if attn_mask_k is not None:
                km = attn_mask_k[:, None, None, :]
                attn_curr = attn_curr * km
                attn_ref = attn_ref * km
            attn_curr = attn_curr / (attn_curr.norm(dim=(-2, -1), keepdim=True) + eps)
            attn_ref = attn_ref / (attn_ref.norm(dim=(-2, -1), keepdim=True) + eps)
            cos = (attn_curr * attn_ref).sum(dim=(-2, -1))  # [B,H]
            penalties.append((1.0 - cos).mean())
    if not penalties:
        return torch.tensor(0.0, device=device)
    return torch.stack(penalties).mean()


def refdir_sinkhorn_penalty_to_ref(
    current_cross: List[torch.Tensor],  # per-layer [B,H,T,S]
    ref_steps: List,  # Either: steps x layers [ [tensor [1,H,1,S]] ] OR reduced layers [tensor [1,S]]
    attn_mask_q: Optional[torch.Tensor],  # [B,T] (per-sample can be [1,T])
    attn_mask_k: Optional[torch.Tensor],  # [B,S] (per-sample can be [1,S])
    blur: float = 0.05,
    margin: float = 0.0,
    layer_stride: int = 1,
    step_mode: str = "all",
    step_stride: int = 1,
    normalize_positions: bool = False,
) -> torch.Tensor:
    """GeomLoss Sinkhorn penalty against per-id reference directory (ALL steps supported).

    Strategy per step & layer:
      1) Average over target dimension: [B,H,T,S] -> [B,H,S]
      2) Average over heads: [B,H,S] -> [B,S]
      3) Apply key mask and normalize to probability on valid src positions
      4) Do same for reference (expand/crop/pad along S to align)
      5) Compute W2 (Sinkhorn) with positions in [0,1]
    """
    try:
        from geomloss import SamplesLoss
    except Exception as e:
        raise ImportError("geomloss is required: pip install geomloss")

    if len(current_cross) == 0 or len(ref_steps) == 0:
        return torch.tensor(
            0.0, device=current_cross[0].device if current_cross else "cpu"
        )

    device = current_cross[0].device
    dtype = current_cross[0].dtype
    eps = 1e-8

    sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=blur, debias=False)

    # Normalize reference format:
    # - If ref_steps is already [steps][layers][tensor ...] keep as is
    # - If ref_steps is [layers][tensor [1,S]], wrap as single step
    if (
        isinstance(ref_steps, list)
        and len(ref_steps) > 0
        and isinstance(ref_steps[0], torch.Tensor)
    ):
        steps_list: List[List[torch.Tensor]] = [ref_steps]  # single pseudo-step
    else:
        steps_list = ref_steps  # type: ignore

    # Select steps
    if step_mode == "all":
        step_indices = range(0, len(steps_list), max(1, step_stride))
    elif step_mode == "first":
        step_indices = [0]
    else:  # last
        step_indices = [len(steps_list) - 1]

    penalties: List[torch.Tensor] = []
    penalties_before_margin: List[float] = []

    # Precompute current averaged per-layer distributions (per-sample assumed B=1 slice)
    # We will align to each step's reference per layer
    layer_indices = range(0, len(current_cross), max(1, layer_stride))
    # print(f"Processing {len(layer_indices)} layers across {len(step_indices)} steps")
    for step_idx in step_indices:
        ref_layers = steps_list[step_idx]
        for layer_idx in layer_indices:
            # print(f"Processing step {step_idx}, layer {layer_idx}")
            if layer_idx >= len(ref_layers):
                continue
            curr = current_cross[layer_idx]  # [B,H,T,S]
            # print(f"curr: {curr.shape}")
            # Average T then H
            curr_avg = curr.mean(dim=2).mean(dim=1)  # [B,S]
            # print(f"curr_avg: {curr_avg.shape}")

            ref = ref_layers[layer_idx]
            # print(f"ref: {ref.shape}")
            ref = ref.to(device=device, dtype=dtype)
            # Support either full [1,H,1,S] or reduced [1,S]
            if ref.dim() == 4:
                # [1,H,1,S]
                ref_avg = ref.mean(dim=1).squeeze(1)  # [1,S]
            elif ref.dim() == 2:
                # [1,S]
                ref_avg = ref
            else:
                # Unexpected shape; skip
                continue

            # Align S dimension by crop/pad on reference to current S
            S = curr_avg.shape[-1]
            S_ref = ref_avg.shape[-1]
            # if S_ref != S:
            # print(f"[DEBUG] S mismatch at layer {layer_idx}: curr={S}, ref={S_ref}")
            if S_ref > S:
                ref_avg = ref_avg[..., :S]
            elif S_ref < S:
                pad = (0, S - S_ref)
                ref_avg = F.pad(ref_avg, pad)

            # Apply masks
            curr_masked = curr_avg
            ref_masked = ref_avg
            if attn_mask_k is not None:
                # broadcast to [B,S]
                mask_k = attn_mask_k
                curr_masked = curr_masked * mask_k
                ref_masked = ref_masked * mask_k[:1]  # reference is [1,S]

            # This is a check for mask
            # print(f"curr_masked: {curr_masked.shape}")
            # print(f"ref_masked: {ref_masked.shape}")
            # print(f"curr_masked: {curr_masked}")
            # print(f"ref_masked: {ref_masked}")
            # Normalize to probability per sample
            curr_norm = curr_masked / (curr_masked.sum(dim=-1, keepdim=True) + eps)
            ref_norm = ref_masked / (ref_masked.sum(dim=-1, keepdim=True) + eps)

            # Build positions: normalized [0,1] or original indices [0,S-1]
            if normalize_positions:
                # Normalized positions: [0, 1]
                positions = torch.linspace(0, 1, S, device=device, dtype=dtype).view(
                    S, 1
                )
            else:
                # Original indices: [0, S-1]
                positions = torch.arange(S, device=device, dtype=dtype).view(S, 1)

            # Compute per-batch Sinkhorn and average
            # reshape weights to [B,S,1] and reference to [1,S,1]
            curr_w = curr_norm.unsqueeze(-1)  # [B,S,1]
            ref_w = ref_norm.unsqueeze(-1)  # [1,S,1]

            # GeomLoss SamplesLoss API: (weights1, positions1, weights2, positions2)
            # We compute per-sample distances and then average
            # SamplesLoss returns a tensor of shape [B] when batched
            X = positions.unsqueeze(0).expand(curr_w.shape[0], -1, -1)  # [B,S,1]
            ref_w_batched = ref_w.expand(curr_w.shape[0], -1, -1)  # [B,S,1]
            w2_sq = sinkhorn_loss(curr_w, X, ref_w_batched, X)
            w2 = torch.sqrt(w2_sq + eps)
            w2_before = w2.mean().item()
            penalties_before_margin.append(w2_before)

            # Apply hard margin: only penalize if distance > margin
            if margin > 0:
                w2 = torch.clamp(w2 - margin, min=0.0)

            # w2_after = w2.mean().item()
            # print(
            #     f"[Sinkhorn] Step {step_idx}, Layer {layer_idx}: before_margin={w2_before:.4f}, after_margin={w2_after:.4f}"
            # )
            penalties.append(w2.mean())

    if not penalties:
        return torch.tensor(0.0, device=device, dtype=dtype), 0.0, 0.0

    penalty_after = torch.stack(penalties).mean()
    avg_before = (
        sum(penalties_before_margin) / len(penalties_before_margin)
        if penalties_before_margin
        else 0.0
    )
    avg_after = penalty_after.item()

    return penalty_after, avg_before, avg_after
