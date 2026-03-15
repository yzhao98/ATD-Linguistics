# ATD-Linguistics

Code for **"Attention Transport Distance Reveals Cross-Lingual Structures in Multilingual Translation Models"**.

This repository provides the full pipeline for computing Attention Transport Distance (ATD) — a metric based on Wasserstein-2 optimal transport distance between cross-attention distributions — to analyze how multilingual translation models organize language knowledge internally.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

The WMT14 newstest French-English reference sentences (3003 lines) are included in `data/wmt.test.fr-en.en`. To regenerate:

```bash
sacrebleu -t wmt14 -l fr-en --echo ref > data/wmt.test.fr-en.en
```

## Pipeline

### Step 1: Extract Attention Matrices

Translate all 3003 sentences to 100 target languages and save cross-attention matrices.

**M2M-100 (encoder-decoder):**

```bash
python -m extraction.extract_m2m_attention
```

**Llama-3.1-8B-Instruct (decoder-only):**

```bash
python -m extraction.extract_llama_attention
```

Output: `{model_dir}/atten_matrix_100/{idx}.pkl` — one file per sentence containing attention matrices for all languages.

### Step 2: Evaluate Translation Quality

Use GPT-4o to score each translation as *yes* (1.0), *almost* (0.5), or *no* (0.0).

```bash
export OPENAI_API_KEY="your-key-here"

# M2M-100
python -m evaluation.eval_quality_m2m

# Llama
python -m evaluation.eval_quality_llama
```

### Step 3: Filter High-Quality Data

Select the top-K sentences by mean quality score and filter languages by quality threshold:

```bash
# M2M-100
python -m evaluation.filter_data --model m2m --top_k 2000 --threshold 0.2

# Llama
python -m evaluation.filter_data --model llama3 --top_k 500 --threshold 0.6
```

Output: `results_m2m/selected_{model}_{top_k}_{threshold}.pkl`

### Step 4: Compute ATD Scores

Compute pairwise Wasserstein-2 distances between attention distributions for all language pairs:

```bash
# M2M-100
python -m distance.cal_m2m_distance

# Llama
python -m distance.cal_llama_distance
```

Output: `results_m2m/all_m2m_distance_dict_fixed_ot.pkl` (or corresponding Llama file)

### Step 5: Visualize Results

**NJ Tree (main visualization):**

```bash
# M2M-100
python visualization/visualize_tree.py --model m2m

# Llama
python visualization/visualize_tree.py --model llama3
```

**Controlled word-order comparison:**

```bash
python visualization/control_stat.py
```

## Transfer Experiment

Fine-tune M2M-100 with attention regularization using sibling language attention patterns as references.

### 1. Download parallel data

```bash
python -m transfer.download_data --src en --tgt ps
python -m transfer.download_data --src en --tgt mr
```

### 2. Precompute sibling attention references

```bash
python transfer/prepare_sibling_attention.py \
    --model_name facebook/m2m100_1.2B \
    --src_lang en \
    --target_lang ps \
    --sibling_langs ar \
    --train_file data/ps_en/train.tsv \
    --output_dir output/ps_penalty/sibling_attention_v3_ar \
    --max_samples 5000
```

### 3. Fine-tune with attention penalty

```bash
python -m transfer.train_with_sibling_penalty \
    --model_name_or_path facebook/m2m100_1.2B \
    --src_lang en --tgt_lang ps \
    --train_file data/ps_en/train.tsv \
    --valid_file data/ps_en/valid.tsv \
    --output_dir transfer/results/ps_ar_sinkhorn \
    --ref_attention_dir output/ps_penalty/sibling_attention_v3_ar \
    --ref_langs ar \
    --lambda_ref 1.0 --lambda_ref_mode relative \
    --ref_sinkhorn --normalize_positions --ref_blur 0.05 \
    --num_train_epochs 20 --lr 3e-5 --seed 42
```


### 4. Evaluate

```bash
python -m transfer.eval_multi_metrics \
    --model_name_or_path facebook/m2m100_1.2B \
    --local_model_path transfer/results/ps_ar_sinkhorn/checkpoint-best \
    --pairs en-ps \
    --output transfer/results/eval_results.json
```

## Project Structure

```
ATD-Linguistics/
├── utils.py                    # Core utilities (language dicts, W2 distance, etc.)
├── requirements.txt
│
├── data/
│   └── wmt.test.fr-en.en      # WMT newstest2014 English (3003 sentences)
│
├── extraction/                 # Step 1: Attention extraction
│   ├── extract_m2m_attention.py
│   └── extract_llama_attention.py
│
├── evaluation/                 # Steps 2-3: Quality evaluation & filtering
│   ├── eval_quality_m2m.py
│   ├── eval_quality_llama.py
│   └── filter_data.py
│
├── distance/                   # Step 4: ATD computation
│   ├── cal_m2m_distance.py
│   └── cal_llama_distance.py
│
├── visualization/              # Step 5: Visualization
│   ├── visualize_tree.py       # NJ tree (--model m2m / llama3)
│   └── control_stat.py
│
└── transfer/                   # Transfer experiment
    ├── train_with_sibling_penalty.py
    ├── eval_multi_metrics.py
    ├── download_data.py
    ├── attention.py
    ├── config.py
    ├── datasets.py
    ├── models.py
    ├── utils.py
    └── prepare_sibling_attention.py
```
