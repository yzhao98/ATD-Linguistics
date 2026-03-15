import random
import re

import numpy as np
import ot
import torch
import openai


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    random.seed(seed)


def get_text_token_positions(prompt, text, tokenizer):
    """Find the token-level start and end positions of `text` within `prompt`."""
    prompt_tokens = tokenizer.tokenize(prompt)
    text_tokens = tokenizer.tokenize(text)

    prompt_token_str = " ".join(prompt_tokens)
    text_token_str = " ".join(text_tokens)

    start_index = prompt_token_str.find(text_token_str)
    if start_index == -1:
        return None, None

    start_token_position = prompt_token_str[:start_index].count(" ")
    end_token_position = start_token_position + len(text_tokens)
    return start_token_position, end_token_position


def accumulate_attention(attentions, prompt_length):
    """
    Accumulate the attention on the prompt part for each generated token.

    Args:
        attentions: list of tensors, each with shape (batch_size, num_heads, seq_len, seq_len)
        prompt_length: int, the length of the prompt

    Returns:
        accumulated_attention: shape (batch_size, num_heads, num_steps, prompt_length)
    """
    batch_size, num_heads, _, seq_len = attentions[0].shape
    total_seq_length = len(attentions)
    accumulated_attention = torch.zeros(
        batch_size, num_heads, total_seq_length, prompt_length
    ).to(attentions[0].device)

    for i, attn in enumerate(attentions):
        current_attention_on_prompt = attn[:, :, :, :prompt_length].clone()
        current_attention_on_previous = attn[:, :, :, -2:-1].clone()
        if i == 0:
            accumulated_attention[:, :, i : i + 1, :] = current_attention_on_prompt
        else:
            accumulated_attention[:, :, i : i + 1, :] = (
                accumulated_attention[:, :, i - 1 : i, :]
                * current_attention_on_previous
                + current_attention_on_prompt
            )

    return accumulated_attention


def extract_translation_with_positions(output, tokenizer):
    """Extract <START>...<END> translation and its token positions."""
    match = re.search(r"<START>(.*?)<END>", output)
    if match:
        translation = match.group(1).strip()
        translation_tokens = tokenizer.tokenize(translation)
        start_index = len(tokenizer.tokenize(output[: match.start(1)]))
        end_index = start_index + len(translation_tokens)
        return translation, start_index, end_index
    return None, None, None


def get_all_lang():
    """Get all 100 target languages supported by M2M-100."""
    tgt_lang_dict = {
        "af": "Afrikaans", "da": "Danish", "nl": "Dutch", "de": "German",
        "en": "English", "is": "Icelandic", "lb": "Luxembourgish", "no": "Norwegian",
        "sv": "Swedish", "fy": "Western Frisian", "yi": "Yiddish",
        "ast": "Asturian", "ca": "Catalan", "fr": "French", "gl": "Galician",
        "it": "Italian", "oc": "Occitan", "pt": "Portuguese", "ro": "Romanian",
        "es": "Spanish",
        "be": "Belarusian", "bs": "Bosnian", "bg": "Bulgarian", "hr": "Croatian",
        "cs": "Czech", "mk": "Macedonian", "pl": "Polish", "ru": "Russian",
        "sr": "Serbian", "sk": "Slovak", "sl": "Slovenian", "uk": "Ukrainian",
        "et": "Estonian", "fi": "Finnish", "hu": "Hungarian",
        "lv": "Latvian", "lt": "Lithuanian",
        "sq": "Albanian", "hy": "Armenian", "ka": "Georgian", "el": "Greek",
        "br": "Breton", "ga": "Irish", "gd": "Scottish Gaelic", "cy": "Welsh",
        "az": "Azerbaijani", "ba": "Bashkir", "kk": "Kazakh", "tr": "Turkish", "uz": "Uzbek",
        "ja": "Japanese", "ko": "Korean", "vi": "Vietnamese", "zh": "Chinese Mandarin",
        "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
        "mr": "Marathi", "ne": "Nepali", "or": "Oriya", "pa": "Panjabi",
        "sd": "Sindhi", "si": "Sinhala", "ur": "Urdu", "ta": "Tamil",
        "ceb": "Cebuano", "ilo": "Iloko", "id": "Indonesian", "jv": "Javanese",
        "mg": "Malagasy", "ms": "Malay", "ml": "Malayalam", "su": "Sundanese", "tl": "Tagalog",
        "my": "Burmese", "km": "Central Khmer", "lo": "Lao", "th": "Thai", "mn": "Mongolian",
        "ar": "Arabic", "he": "Hebrew", "ps": "Pashto", "fa": "Farsi",
        "am": "Amharic", "ff": "Fulah", "ha": "Hausa", "ig": "Igbo",
        "ln": "Lingala", "lg": "Luganda", "ns": "Northern Sotho",
        "so": "Somali", "sw": "Swahili", "ss": "Swati", "tn": "Tswana",
        "wo": "Wolof", "xh": "Xhosa", "yo": "Yoruba", "zu": "Zulu",
        "ht": "Haitian Creole",
    }
    all_tgt_lang_list = list(tgt_lang_dict.keys())
    return tgt_lang_dict, all_tgt_lang_list


def compute_w2_distance(input_accumulated_atten_dict, lang1, lang2, layer):
    """Compute Wasserstein-2 distance between attention distributions of two languages at a given layer."""
    a = np.array(input_accumulated_atten_dict[lang1][layer])
    b = np.array(input_accumulated_atten_dict[lang2][layer])
    assert a.shape == b.shape, "Vectors must have the same shape."

    wa = a / a.sum()
    wb = b / b.sum()
    support = np.arange(len(a)).reshape(-1, 1)
    M = ot.dist(support, support, metric="sqeuclidean")
    w2_squared = ot.emd2(wa, wb, M)
    w2 = np.sqrt(w2_squared)
    return w2


def get_remove_list():
    """Get the list of languages to be removed from analysis."""
    return [
        "br", "ga", "gd", "uz", "ne", "or", "pa", "my", "ff", "ln", "lg", "ns", "so", "tn", "wo",
    ]


def eval_response_by_llm(original_text, translated_text, tgt_lang, model="gpt-4o"):
    """Use GPT-4o to evaluate translation quality. Returns 'yes', 'almost', or 'no'."""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert translation evaluator. "
                    "Classify the quality of a translation based on the following rules:\n\n"
                    "- Respond 'yes' if the translation is in the correct target language, is grammatically readable, "
                    "and the meaning is correct—even if there are small grammar or word choice issues.\n"
                    "- Respond 'almost' if the translation is in the correct target language but includes incorrect key words, "
                    "semantic mistakes, or seems like a direct word-for-word substitution.\n"
                    "- Respond 'no' if the translation is in the wrong language, completely incoherent, grammatically broken "
                    "beyond repair, or fails to resemble a real sentence.\n"
                    "- Check whether the translation is written in the correct target language. If not, respond 'no'.\n"
                    "Only reply with one word: yes, almost, or no."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original English text: {original_text}\n"
                    f"Translated text in {tgt_lang}: {translated_text}\n"
                    f"What is your evaluation? Respond with one word only: yes, almost, or no."
                ),
            },
        ],
        max_tokens=10,
        timeout=60,
    )
    answer = response.choices[0].message.content.strip().lower()
    return answer


# =====================================================================
# Language Family Classifications
# =====================================================================

languages_by_family = {
    "Indo-European": {
        "Romance": ["es", "pt", "gl", "ca", "fr", "it", "ro", "oc", "ast"],
        "Germanic": ["en", "de", "nl", "af", "fy", "lb", "is", "sv", "da", "no"],
        "Slavic": ["ru", "pl", "cs", "sk", "sl", "hr", "sr", "mk", "bg", "uk", "bs"],
        "Baltic": ["lt", "lv"],
        "Hellenic": ["el"],
        "Albanian": ["sq"],
        "Indo-Aryan": ["hi", "ur", "bn", "si"],
        "Iranian": ["fa"],
    },
    "Uralic": {"Finnic_Ugric": ["fi", "et", "hu"]},
    "Austronesian": {"Malayo-Polynesian": ["id", "ms", "jv", "su", "mg"]},
    "Austroasiatic": {"Vietic": ["vi"]},
    "Sino-Tibetan": {"Chinese": ["zh"]},
    "Tai-Kadai": {"Kam-Tai": ["th"]},
    "Afro-Asiatic": {"Semitic": ["he"]},
    "Isolates": {"Japanese": ["ja"], "Korean": ["ko"]},
    "Turkic": {"Oghuz": ["tr"]},
    "Creole": {"French-based": ["ht"]},
    "Niger-Congo": {"Bantu": ["sw"]},
}

lang_to_family = {
    "es": ("Indo-European", "Romance"), "pt": ("Indo-European", "Romance"),
    "gl": ("Indo-European", "Romance"), "ca": ("Indo-European", "Romance"),
    "fr": ("Indo-European", "Romance"), "it": ("Indo-European", "Romance"),
    "ro": ("Indo-European", "Romance"), "oc": ("Indo-European", "Romance"),
    "ast": ("Indo-European", "Romance"),
    "en": ("Indo-European", "Germanic"), "de": ("Indo-European", "Germanic"),
    "nl": ("Indo-European", "Germanic"), "af": ("Indo-European", "Germanic"),
    "fy": ("Indo-European", "Germanic"), "lb": ("Indo-European", "Germanic"),
    "is": ("Indo-European", "Germanic"), "sv": ("Indo-European", "Germanic"),
    "da": ("Indo-European", "Germanic"), "no": ("Indo-European", "Germanic"),
    "ru": ("Indo-European", "Slavic"), "pl": ("Indo-European", "Slavic"),
    "cs": ("Indo-European", "Slavic"), "sk": ("Indo-European", "Slavic"),
    "sl": ("Indo-European", "Slavic"), "hr": ("Indo-European", "Slavic"),
    "sr": ("Indo-European", "Slavic"), "mk": ("Indo-European", "Slavic"),
    "bg": ("Indo-European", "Slavic"), "uk": ("Indo-European", "Slavic"),
    "bs": ("Indo-European", "Slavic"),
    "lt": ("Indo-European", "Baltic"), "lv": ("Indo-European", "Baltic"),
    "el": ("Indo-European", "Hellenic"), "sq": ("Indo-European", "Albanian"),
    "hi": ("Indo-European", "Indo-Aryan"), "ur": ("Indo-European", "Indo-Aryan"),
    "bn": ("Indo-European", "Indo-Aryan"), "si": ("Indo-European", "Indo-Aryan"),
    "fa": ("Indo-European", "Iranian"),
    "fi": ("Uralic", "Finnic-Ugric"), "et": ("Uralic", "Finnic-Ugric"),
    "hu": ("Uralic", "Finnic-Ugric"),
    "id": ("Austronesian", "Malayo-Polynesian"), "ms": ("Austronesian", "Malayo-Polynesian"),
    "jv": ("Austronesian", "Malayo-Polynesian"), "su": ("Austronesian", "Malayo-Polynesian"),
    "mg": ("Austronesian", "Malayo-Polynesian"),
    "vi": ("Austroasiatic", "Vietic"),
    "zh": ("Sino-Tibetan", "Chinese"),
    "th": ("Tai-Kadai", "Kam-Tai"),
    "he": ("Afro-Asiatic", "Semitic"),
    "ja": ("Japonic", "Japanese"), "ko": ("Koreanic", "Korean"),
    "tr": ("Turkic", "Oghuz"),
    "ht": ("Creole", "French-based"),
    "sw": ("Niger-Congo", "Bantu"),
}

# Indo-European tree structure for phylogenetic analysis
IE_TREE = (
    "Indo-European",
    [
        ("Romance", [
            ("Western Romance", [
                ("Gallo-Romance", ["fr", "oc"]),
                ("Iberian Romance", ["pt", "gl", "es", "ast", "ca"]),
            ]),
            ("Italo-Romance", ["it", "ro"]),
        ]),
        ("Germanic", [
            ("West Germanic", ["de", "nl", "af", "fy", "lb", "en"]),
            ("North Germanic", ["da", "sv", "no", "is"]),
        ]),
        ("Balto-Slavic", [
            ("Baltic", ["lv", "lt"]),
            ("Slavic", [
                ("East Slavic", ["ru", "uk"]),
                ("West Slavic", ["pl", "cs", "sk"]),
                ("South Slavic", ["bg", "mk", "sr", "hr", "sl", "bs"]),
            ]),
        ]),
        ("Hellenic", ["el"]),
        ("Albanian", ["sq"]),
        ("Armenian", ["hy"]),
        ("Celtic", [
            ("Goidelic", ["ga", "gd"]),
            ("Brythonic", ["cy", "br"]),
        ]),
        ("Indo-Iranian", [
            ("Indo-Aryan", ["hi", "ur", "bn", "si"]),
            ("Iranian", ["fa"]),
        ]),
    ],
)


def ie_tree_leaves(tree=IE_TREE):
    """Extract all leaf language codes from the IE tree."""
    label, children = tree
    leaves = []
    for child in children:
        if isinstance(child, tuple):
            leaves.extend(ie_tree_leaves(child))
        elif isinstance(child, list):
            leaves.extend(child)
    return leaves


# Languages to optionally exclude when building NJ trees
DELETED_LANG = []
DELETED_LANG_LLAMA = ["fa"]


def torch_w2_emd_squared_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    1D Wasserstein-2 distance with squared Euclidean cost on integer support {0,...,N-1}.
    Uses the exact monotone transport algorithm.

    Args:
        p, q: [B, N] nonnegative tensors (not necessarily normalized)

    Returns:
        w2: [B] tensor of W2 distances
    """
    eps = 1e-12
    device = p.device
    p = p.clamp(min=0)
    q = q.clamp(min=0)
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    q = q / (q.sum(dim=-1, keepdim=True) + eps)
    B, N = p.shape
    p_np = p.detach().cpu().numpy()
    q_np = q.detach().cpu().numpy()
    out = torch.zeros((B,), dtype=p.dtype, device=device)
    for b in range(B):
        pi = p_np[b].copy()
        qi = q_np[b].copy()
        i = j = 0
        cost = 0.0
        while i < N and j < N:
            flow = min(pi[i], qi[j])
            if flow > 0:
                dij = i - j
                cost += flow * (dij * dij)
                pi[i] -= flow
                qi[j] -= flow
            if pi[i] <= 1e-18:
                i += 1
            if qi[j] <= 1e-18:
                j += 1
        out[b] = float(cost) ** 0.5
    return out


def compute_cophenetic_correlation(tree_root, original_dist_matrix, labels):
    """
    Compute Cophenetic Correlation Coefficient to measure NJ tree fidelity.

    Args:
        tree_root: NJ tree root node
        original_dist_matrix: original distance matrix (n x n)
        labels: language labels (matching distance matrix order)

    Returns:
        dict with pearson_r, pearson_p, spearman_r, spearman_p, tree_dist_matrix, tip_names
    """
    from scipy.stats import pearsonr, spearmanr

    tips = list(tree_root.tips())
    n = len(tips)
    tip_names = [t.name for t in tips]

    def get_path_to_root(node):
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    paths = {id(tip): get_path_to_root(tip) for tip in tips}

    tree_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            path_i = paths[id(tips[i])]
            path_j = paths[id(tips[j])]
            set_i = set(id(node) for node in path_i)
            lca = None
            for node in path_j:
                if id(node) in set_i:
                    lca = node
                    break
            dist = 0.0
            for tip, path in [(tips[i], path_i), (tips[j], path_j)]:
                for node in path:
                    if node == lca:
                        break
                    dist += node.length
            tree_dist[i, j] = dist
            tree_dist[j, i] = dist

    idx_map = [labels.index(name) for name in tip_names]
    original_reordered = original_dist_matrix[np.ix_(idx_map, idx_map)]
    upper_tri = np.triu_indices(n, k=1)
    tree_dists = tree_dist[upper_tri]
    orig_dists = original_reordered[upper_tri]

    pearson_r, pearson_p = pearsonr(tree_dists, orig_dists)
    spearman_r, spearman_p = spearmanr(tree_dists, orig_dists)

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "tree_dist_matrix": tree_dist,
        "tip_names": tip_names,
    }


def print_cophenetic_correlation(tree_root, original_dist_matrix, labels, method_name=""):
    """Compute and print Cophenetic Correlation results."""
    result = compute_cophenetic_correlation(tree_root, original_dist_matrix, labels)

    prefix = f"[{method_name}] " if method_name else ""
    print(f"\n{'=' * 60}")
    print(f"{prefix}Cophenetic Correlation (NJ Tree Fidelity)")
    print(f"{'=' * 60}")
    print(f"  Pearson r:  {result['pearson_r']:.4f}  (p = {result['pearson_p']:.2e})")
    print(f"  Spearman r: {result['spearman_r']:.4f}  (p = {result['spearman_p']:.2e})")
    print(f"{'=' * 60}")

    r = result["pearson_r"]
    if r >= 0.9:
        quality = "Excellent - tree preserves original distances well"
    elif r >= 0.8:
        quality = "Good - tree reasonably preserves original distances"
    elif r >= 0.7:
        quality = "Acceptable - some information loss"
    else:
        quality = "Poor - data may not be well suited for tree representation"
    print(f"  Interpretation: {quality}")
    print(f"{'=' * 60}\n")
    return result


def torch_w1_emd_1d(p: torch.Tensor, q: torch.Tensor, positions: str = "int") -> torch.Tensor:
    """
    1D Wasserstein-1 distance with uniform bins.

    Args:
        positions: "int" for support at 0..N-1, "unit" for linspace(0,1,N)
    """
    eps = 1e-12
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    q = q / (q.sum(dim=-1, keepdim=True) + eps)
    diff = torch.cumsum(p - q, dim=-1).abs()
    if positions == "unit":
        N = p.size(-1)
        return diff.sum(dim=-1) * (1.0 / max(1, N - 1))
    return diff.sum(dim=-1)


# =====================================================================
# Language Word Order Classification
# =====================================================================

LANGUAGE_WORD_ORDER = {
    "SVO": [
        "en", "de", "nl", "af", "da", "no", "sv", "is", "lb", "fy", "yi",
        "fr", "es", "pt", "it", "ro", "ca", "gl", "oc", "ast", "ht",
        "ru", "uk", "be", "pl", "cs", "sk", "bg", "mk", "sr", "hr", "bs", "sl",
        "lt", "lv", "el", "sq", "hy", "fi", "et", "hu",
        "id", "ms", "jv", "su", "vi", "km", "th", "lo", "zh",
        "sw", "yo", "ig", "zu", "xh", "ln", "lg", "wo", "ff", "ns", "ss", "tn", "ha",
    ],
    "SOV": [
        "ja", "ko", "tr", "az", "uz", "kk", "ba", "mn",
        "hi", "ur", "bn", "pa", "gu", "mr", "ne", "si", "sd", "or",
        "fa", "ps", "ta", "kn", "ml", "am", "so", "ka", "my",
    ],
    "VSO": ["ar", "he", "ga", "gd", "cy", "br", "tl", "ceb", "ilo"],
    "VOS": ["mg"],
}


def get_word_order_for_lang(lang_code):
    """Get the word order type for a given language code."""
    for word_order, langs in LANGUAGE_WORD_ORDER.items():
        if lang_code in langs:
            return word_order
    return "Unknown"


def get_langs_by_word_order(word_order):
    """Get all languages with a specific word order."""
    return LANGUAGE_WORD_ORDER.get(word_order, [])


# =====================================================================
# Controlled Comparison: Focal Languages and Their Groupings
# =====================================================================

FOCAL_LANGUAGE_GROUPS = {
    "ja": {
        "name": "Japanese",
        "word_order": "SOV",
        "related_languages": ["ko", "zh"],
        "exclusion_reason": (
            "Korean shares extensive areal features and similar structure; "
            "Chinese has strong historical contact and influence on Japanese"
        ),
    },
    "xh": {
        "name": "Xhosa",
        "word_order": "SVO",
        "related_languages": ["ss", "zu", "sw", "yo", "ig", "af", "nl"],
        "exclusion_reason": (
            "Swati and Zulu are closely related Nguni languages; "
            "Swahili/Yoruba/Igbo are Niger-Congo family; "
            "Afrikaans/Dutch have areal contact in South Africa"
        ),
    },
    "vi": {
        "name": "Vietnamese",
        "word_order": "SVO",
        "related_languages": ["zh", "km", "lo", "th", "fr"],
        "exclusion_reason": (
            "Chinese has massive lexical influence; "
            "Khmer is Austroasiatic family; "
            "Lao/Thai share MSEA Sprachbund features; "
            "French has colonial influence"
        ),
    },
}


def get_focal_language_groups():
    """Get the focal language comparison groups."""
    return FOCAL_LANGUAGE_GROUPS


def get_comparison_groups_for_focal(focal_lang, available_langs=None):
    """
    Get comparison groups (same/different word order) for a focal language,
    excluding genetically/areally related languages.
    """
    if focal_lang not in FOCAL_LANGUAGE_GROUPS:
        raise ValueError(f"Unknown focal language: {focal_lang}")

    info = FOCAL_LANGUAGE_GROUPS[focal_lang]
    focal_word_order = info["word_order"]
    related = set(info["related_languages"])

    if available_langs is None:
        all_langs = set()
        for word_order_langs in LANGUAGE_WORD_ORDER.values():
            all_langs.update(word_order_langs)
        available_langs = all_langs
    else:
        available_langs = set(available_langs)

    exclude_set = related | {focal_lang}
    remaining_langs = available_langs - exclude_set

    same_wo = []
    diff_wo = []
    for lang in remaining_langs:
        lang_wo = get_word_order_for_lang(lang)
        if lang_wo == focal_word_order:
            same_wo.append(lang)
        elif lang_wo != "Unknown":
            diff_wo.append(lang)

    return {
        "focal": focal_lang,
        "focal_name": info["name"],
        "focal_word_order": focal_word_order,
        "excluded": sorted(list(related & available_langs)),
        "excluded_reason": info["exclusion_reason"],
        "same_word_order": sorted(same_wo),
        "diff_word_order": sorted(diff_wo),
    }
