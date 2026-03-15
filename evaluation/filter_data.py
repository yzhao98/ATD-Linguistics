"""
Filter high-quality sentences and languages based on GPT-4o evaluation scores.

Selects the top-K sentences by mean quality score, then filters languages
by a quality threshold. Outputs selected indices and languages as a pickle file.

Usage:
    python -m evaluation.filter_data --model m2m --top_k 2000 --threshold 0.2
    python -m evaluation.filter_data --model llama3 --top_k 500 --threshold 0.6
"""

import argparse
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="m2m", help="Model name (m2m or llama3)")
    parser.add_argument("--top_k", type=int, default=2000, help="Number of top sentences to select")
    parser.add_argument("--threshold", type=float, default=0.0, help="Min language quality threshold")
    args = parser.parse_args()

    model_name = args.model
    OUT_DIR = f"results_{model_name}"

    top_k = args.top_k
    threshold = args.threshold

    # Read quality evaluation CSV
    df = pd.read_csv(f"{OUT_DIR}/{model_name}_trans_llm_eval_all.csv")
    df[df < 0] = 0
    print(df.shape)

    # Compute mean score per sentence across all languages
    index_mean = df.mean(axis=1)

    # Plot distribution of mean scores
    plt.figure(figsize=(10, 6))
    plt.hist(index_mean, bins=50)
    plt.xlabel("Mean Score per Index")
    plt.ylabel("Frequency")
    plt.title("Distribution of Mean Scores Across Indexes")
    plt.grid(True)
    plt.show()

    # Select top-K sentences by mean quality
    top_indices = index_mean.nlargest(top_k).index

    # Compute per-language mean on selected sentences
    top_df = df.loc[top_indices]
    language_mean = top_df.mean(axis=0)

    # Plot per-language quality
    plt.figure(figsize=(12, 6))
    language_mean.sort_values(ascending=False).plot(kind="bar")
    plt.ylabel("Mean Score")
    plt.title(f"Mean Score per Language for Top {top_k} Indexes")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Filter languages by threshold
    selected_languages = language_mean[language_mean > threshold].index
    selected_indices = top_df.index
    selected_df = top_df[selected_languages]
    print(selected_df.shape)

    # Remove English (source language)
    if "en" in selected_languages:
        selected_languages = selected_languages.drop("en")
    if "en" in selected_df.columns:
        selected_df = selected_df.drop(columns=["en"])

    selected_indices_list = selected_indices.tolist()
    selected_languages_list = selected_languages.tolist()

    # Plot heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(selected_df, cmap="viridis", cbar_kws={"label": "Score"})
    plt.title("Heatmap of Selected Indexes and Languages")
    plt.xlabel("Language")
    plt.ylabel("Index")
    plt.tight_layout()
    plt.show()

    # Save selected indices and languages
    selected_data = {
        "selected_indices": selected_indices_list,
        "selected_languages": selected_languages_list,
    }
    with open(f"{OUT_DIR}/selected_{model_name}_{top_k}_{threshold}.pkl", "wb") as f:
        pickle.dump(selected_data, f)

    # Plot selected language quality distribution
    selected_language_means = language_mean[selected_languages]
    plt.figure(figsize=(15, 6))
    selected_language_means.sort_values(ascending=False).plot(kind="bar")
    plt.ylabel("Mean Score")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}/mean_score_selected_languages_{model_name}_{top_k}_{threshold}.png",
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":
    main()
