import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import sacrebleu
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:  # fallback if tqdm unavailable
    tqdm = None  # type: ignore

from .config import INDO_EUROPEAN_PAIRS_DEFAULT, M2M_CODE_MAP
from .datasets import ParallelTextDataset
from .models import load_model_and_tokenizer, set_m2m_langs
from .utils import ensure_dir, save_json


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate multiple language pairs with BLEU/chrF/TER/COMET and basic stats"
    )
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_model_path", type=str, default="")
    p.add_argument(
        "--pairs",
        type=str,
        nargs="*",
        default=[],
        help="Pairs like hi-en mr-en; defaults to IE set if empty or --ie_only",
    )
    p.add_argument(
        "--ie_only", action="store_true", help="Use built-in Indo-European pairs"
    )
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--limit", type=int, default=0, help="Evaluate only first N examples (0 = all)"
    )
    # COMET (optional)
    p.add_argument(
        "--comet_model",
        type=str,
        default="",
        help="e.g., Unbabel/wmt22-comet-da; empty to skip",
    )
    p.add_argument("--comet_batch_size", type=int, default=32)
    p.add_argument("--output", type=str, default="results/eval_multi_metrics.json")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


def load_split_path(data_root: str, src: str, tgt: str) -> Tuple[str, str]:
    folder = f"{tgt}_{src}" if src == "en" else f"{src}_{tgt}"
    test_path = os.path.join(data_root, folder, "test.tsv")
    if not os.path.exists(test_path):
        test_path = os.path.join(data_root, folder, "valid.tsv")
    return test_path, folder


def compute_stats(
    sys_out: List[str], refs: List[str], srcs: List[str]
) -> Dict[str, float]:
    def avg_len(texts: List[str]) -> float:
        if not texts:
            return 0.0
        return sum(len(t.split()) for t in texts) / len(texts)

    exact = 0
    for h, r in zip(sys_out, refs):
        if h.strip() == r.strip():
            exact += 1
    return {
        "avg_src_len": avg_len(srcs),
        "avg_hyp_len": avg_len(sys_out),
        "avg_ref_len": avg_len(refs),
        "length_ratio": (avg_len(sys_out) / max(1e-9, avg_len(srcs))) if srcs else 0.0,
        "exact_match_pct": (100.0 * exact / max(1, len(refs))) if refs else 0.0,
    }


def maybe_load_comet(model_name: str):
    if not model_name:
        return None
    try:
        from comet import download_model, load_from_checkpoint  # type: ignore
    except Exception as e:
        print(f"COMET not available ({e}); skipping COMET")
        return None
    ckpt_path = download_model(model_name)
    model = load_from_checkpoint(ckpt_path)
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.ie_only or not args.pairs:
        pairs: List[Tuple[str, str]] = INDO_EUROPEAN_PAIRS_DEFAULT
    else:
        pairs = [tuple(p.split("-")) for p in args.pairs]

    # Normalize direction: if English is present, force src='en', tgt='other'
    norm_pairs: List[Tuple[str, str]] = []
    for src, tgt in pairs:
        if src == "en" or tgt == "en":
            other = tgt if src == "en" else src
            norm_pairs.append(("en", other))
        else:
            norm_pairs.append((src, tgt))

    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path, args.local_model_path, fp16=args.fp16
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_results: Dict[str, Dict[str, float]] = {}
    comet_model = maybe_load_comet(args.comet_model)

    for src, tgt in norm_pairs:
        src_code = M2M_CODE_MAP.get(src, src)
        tgt_code = M2M_CODE_MAP.get(tgt, tgt)
        set_m2m_langs(model, tokenizer, src_code, tgt_code)
        path, _ = load_split_path(args.data_root, src, tgt)
        if not os.path.exists(path):
            continue
        dataset = ParallelTextDataset(path)
        if args.limit and args.limit > 0:
            dataset.examples = dataset.examples[: args.limit]
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        sys_out: List[str] = []
        refs: List[str] = []
        srcs: List[str] = []

        iterable = loader
        total_batches = None
        if hasattr(loader, "__len__"):
            try:
                total_batches = len(loader)
            except Exception:
                total_batches = None
        if tqdm is not None:
            desc = f"Evaluating {src}->{tgt}"
            iterable = tqdm(loader, total=total_batches, desc=desc)

        for batch in iterable:
            inputs = tokenizer(
                batch["src"],
                padding=True,
                truncation=False,
                return_tensors="pt",
            ).to(device)
            gen = model.generate(
                **inputs,
                num_beams=args.num_beams,
                return_dict_in_generate=True,
                output_attentions=False,
            )
            sequences = gen.sequences if hasattr(gen, "sequences") else gen
            decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            sys_out.extend(decoded)
            refs.extend(batch["tgt"])
            srcs.extend(batch["src"])

        pair_key = f"{src}->{tgt}"
        if refs:
            bleu = sacrebleu.corpus_bleu(sys_out, [refs]).score
            chrf = sacrebleu.corpus_chrf(sys_out, [refs]).score
            ter = sacrebleu.corpus_ter(sys_out, [refs]).score
            stats = compute_stats(sys_out, refs, srcs)
            all_results[pair_key] = {
                "sacreBLEU": bleu,
                "chrF": chrf,
                "TER": ter,
                "n": float(len(refs)),
                **stats,
            }
            if comet_model is not None:
                data = [
                    {"src": s, "mt": h, "ref": r}
                    for s, h, r in zip(srcs, sys_out, refs)
                ]
                try:
                    gpus = 1 if torch.cuda.is_available() else 0
                    output = comet_model.predict(
                        data, batch_size=args.comet_batch_size, gpus=gpus
                    )
                    system_score = (
                        float(output["system_score"])
                        if isinstance(output, dict)
                        else float(getattr(output, "system_score", 0.0))
                    )
                    all_results[pair_key]["COMET"] = system_score
                except Exception as e:
                    print(f"COMET failed for {pair_key}: {e}")
                    all_results[pair_key]["COMET"] = 0.0
        else:
            all_results[pair_key] = {
                "sacreBLEU": 0.0,
                "chrF": 0.0,
                "TER": 0.0,
                "n": 0.0,
            }

    ensure_dir(os.path.dirname(args.output) or ".")
    save_json(all_results, args.output)

    print("\nPair\tBLEU\tchrF\tTER\tCOMET\tN")
    for k, v in all_results.items():
        print(
            f"{k}\t{v.get('sacreBLEU',0):.2f}\t{v.get('chrF',0):.2f}\t{v.get('TER',0):.2f}\t{v.get('COMET',0):.3f}\t{int(v.get('n',0))}"
        )


if __name__ == "__main__":
    main()
