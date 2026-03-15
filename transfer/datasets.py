from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple

import torch
from torch.utils.data import Dataset


def _read_tsv(path: str) -> List[Tuple[str, str]]:
    import gzip

    open_fn = gzip.open if path.endswith(".gz") else open
    pairs: List[Tuple[str, str]] = []
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pairs.append((parts[0], parts[1]))
    return pairs


def _read_jsonl(path: str) -> List[Tuple[str, str]]:
    import gzip
    import json

    open_fn = gzip.open if path.endswith(".gz") else open
    pairs: List[Tuple[str, str]] = []
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            src = obj.get("src")
            tgt = obj.get("tgt")
            if src is None or tgt is None:
                continue
            pairs.append((src, tgt))
    return pairs


class ParallelTextDataset(Dataset):
    def __init__(self, path: str):
        if path.endswith(".tsv") or path.endswith(".tsv.gz"):
            self.examples = _read_tsv(path)
        else:
            self.examples = _read_jsonl(path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        s, t = self.examples[idx]
        return {"src": s, "tgt": t, "sample_idx": idx}


@dataclass
class DataCollatorSeq2Seq:
    tokenizer: any
    max_source_length: int
    max_target_length: int

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        sources = [b["src"] for b in batch]
        targets = [b["tgt"] for b in batch]
        model_inputs = self.tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                padding=True,
                truncation=True,
                max_length=self.max_target_length,
                return_tensors="pt",
            )
        model_inputs["labels"] = labels["input_ids"].masked_fill(
            labels["input_ids"] == self.tokenizer.pad_token_id, -100
        )
        model_inputs["sample_idx"] = torch.tensor(
            [b["sample_idx"] for b in batch], dtype=torch.long
        )
        return model_inputs
