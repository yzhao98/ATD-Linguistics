"""
NJ Tree Visualization for ATD Results.

Generates NJ tree, geographic scatter maps, heatmaps, and IE tree plots.
Supports both M2M-100 and Llama-3.1-8B models.

Usage:
    python visualization/visualize_tree.py --model m2m
    python visualization/visualize_tree.py --model llama3
"""

import argparse
import os
import sys
import pickle
import colorsys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

from utils import print_cophenetic_correlation


# =====================================================================
# Language data
# =====================================================================


def get_all_lang():
    """Get all languages"""
    tgt_lang_dict = {
        "af": "Afrikaans",
        "da": "Danish",
        "nl": "Dutch",
        "de": "German",
        "en": "English",
        "is": "Icelandic",
        "lb": "Luxembourgish",
        "no": "Norwegian",
        "sv": "Swedish",
        "fy": "Western Frisian",
        "yi": "Yiddish",
        "ast": "Asturian",
        "ca": "Catalan",
        "fr": "French",
        "gl": "Galician",
        "it": "Italian",
        "oc": "Occitan",
        "pt": "Portuguese",
        "ro": "Romanian",
        "es": "Spanish",
        "be": "Belarusian",
        "bs": "Bosnian",
        "bg": "Bulgarian",
        "hr": "Croatian",
        "cs": "Czech",
        "mk": "Macedonian",
        "pl": "Polish",
        "ru": "Russian",
        "sr": "Serbian",
        "sk": "Slovak",
        "sl": "Slovenian",
        "uk": "Ukrainian",
        "et": "Estonian",
        "fi": "Finnish",
        "hu": "Hungarian",
        "lv": "Latvian",
        "lt": "Lithuanian",
        "sq": "Albanian",
        "hy": "Armenian",
        "ka": "Georgian",
        "el": "Greek",
        "br": "Breton",
        "ga": "Irish",
        "gd": "Scottish Gaelic",
        "cy": "Welsh",
        "az": "Azerbaijani",
        "ba": "Bashkir",
        "kk": "Kazakh",
        "tr": "Turkish",
        "uz": "Uzbek",
        "ja": "Japanese",
        "ko": "Korean",
        "vi": "Vietnamese",
        "zh": "Chinese",
        "bn": "Bengali",
        "gu": "Gujarati",
        "hi": "Hindi",
        "kn": "Kannada",
        "mr": "Marathi",
        "ne": "Nepali",
        "or": "Oriya",
        "pa": "Panjabi",
        "sd": "Sindhi",
        "si": "Sinhala",
        "ur": "Urdu",
        "ta": "Tamil",
        "ceb": "Cebuano",
        "ilo": "Iloko",
        "id": "Indonesian",
        "jv": "Javanese",
        "mg": "Malagasy",
        "ms": "Malay",
        "ml": "Malayalam",
        "su": "Sundanese",
        "tl": "Tagalog",
        "my": "Burmese",
        "km": "Central Khmer",
        "lo": "Lao",
        "th": "Thai",
        "mn": "Mongolian",
        "ar": "Arabic",
        "he": "Hebrew",
        "ps": "Pashto",
        "fa": "Farsi",
        "am": "Amharic",
        "ff": "Fulah",
        "ha": "Hausa",
        "ig": "Igbo",
        "ln": "Lingala",
        "lg": "Luganda",
        "ns": "Northern Sotho",
        "so": "Somali",
        "sw": "Swahili",
        "ss": "Swati",
        "tn": "Tswana",
        "wo": "Wolof",
        "xh": "Xhosa",
        "yo": "Yoruba",
        "zu": "Zulu",
        "ht": "Haitian Creole",
    }
    return tgt_lang_dict, list(tgt_lang_dict.keys())


# Language family mapping
language_family_map = {
    # Indo-European (IE)
    "en": "IE",
    "de": "IE",
    "nl": "IE",
    "af": "IE",
    "da": "IE",
    "no": "IE",
    "sv": "IE",
    "is": "IE",
    "lb": "IE",
    "fy": "IE",
    "yi": "IE",
    "fr": "IE",
    "es": "IE",
    "pt": "IE",
    "it": "IE",
    "ro": "IE",
    "ca": "IE",
    "gl": "IE",
    "oc": "IE",
    "ast": "IE",
    "ht": "IE",
    "ru": "IE",
    "uk": "IE",
    "be": "IE",
    "pl": "IE",
    "cs": "IE",
    "sk": "IE",
    "bg": "IE",
    "mk": "IE",
    "sr": "IE",
    "hr": "IE",
    "bs": "IE",
    "sl": "IE",
    "lt": "IE",
    "lv": "IE",
    "ga": "IE",
    "gd": "IE",
    "cy": "IE",
    "br": "IE",
    "el": "IE",
    "sq": "IE",
    "hy": "IE",
    "hi": "IE",
    "ur": "IE",
    "bn": "IE",
    "pa": "IE",
    "gu": "IE",
    "mr": "IE",
    "ne": "IE",
    "si": "IE",
    "sd": "IE",
    "or": "IE",
    "fa": "IE",
    "ps": "IE",
    # Uralic (UR)
    "fi": "UR",
    "et": "UR",
    "hu": "UR",
    # Turkic (TK)
    "tr": "TK",
    "az": "TK",
    "uz": "TK",
    "kk": "TK",
    "ba": "TK",
    # Afro-Asiatic (AA)
    "ar": "AA",
    "he": "AA",
    "am": "AA",
    "so": "AA",
    "ha": "AA",
    # Sino-Tibetan (ST)
    "zh": "ST",
    "my": "ST",
    # Japonic (JP)
    "ja": "JP",
    # Koreanic (KR)
    "ko": "KR",
    # Austronesian (AN)
    "id": "AN",
    "ms": "AN",
    "tl": "AN",
    "ceb": "AN",
    "ilo": "AN",
    "jv": "AN",
    "su": "AN",
    "mg": "AN",
    # Austroasiatic (AS)
    "vi": "AS",
    "km": "AS",
    # Tai-Kadai (TAI)
    "th": "TAI",
    "lo": "TAI",
    # Dravidian (DR)
    "ta": "DR",
    "kn": "DR",
    "ml": "DR",
    # Kartvelian (KA)
    "ka": "KA",
    # Mongolic (MO)
    "mn": "MO",
    # Niger-Congo (NC)
    "sw": "NC",
    "yo": "NC",
    "ig": "NC",
    "zu": "NC",
    "xh": "NC",
    "ln": "NC",
    "lg": "NC",
    "wo": "NC",
    "ff": "NC",
    "ns": "NC",
    "ss": "NC",
    "tn": "NC",
}


# =====================================================================
# Model-specific configuration
# =====================================================================

MODEL_CONFIGS = {
    "m2m": {
        "top_k": 2000,
        "threshold": 0.2,
        "results_dir": "results_m2m",
        "suffix": "_fixed_ot",
        "tree_figsize": (20, 24),
        "tree_fontsize": 14,
        "ht_family": "IE+NC Creole",
    },
    "llama3": {
        "top_k": 500,
        "threshold": 0.6,
        "results_dir": "results_llama3",
        "suffix": "_fixed_clip_layer",
        "tree_figsize": (24, 24),
        "tree_fontsize": 18,
        "ht_family": "IE",
    },
}

# Number of clusters to generate
N_CLUSTERS_LIST = [7, 10]
MAX_SUBCLUSTERS = 3

# Eurasia-Africa map extent
EURASIA_EXTENT = [-25, 150, -40, 75]
# Extended extent (includes Haiti, tighter latitude range)
EURASIA_EXTENDED_EXTENT = [-80, 150, -35, 70]


# =====================================================================
# NJ tree node class
# =====================================================================


class NJTreeNode:
    """NJ tree node"""

    def __init__(self, name, length=0.0):
        self.name = name
        self.length = length
        self.children = []
        self.parent = None

    def is_tip(self):
        return len(self.children) == 0

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def tips(self):
        """Return all leaf nodes"""
        if self.is_tip():
            yield self
        else:
            for child in self.children:
                yield from child.tips()

    def to_newick(self):
        """Convert to Newick format"""
        if self.is_tip():
            return f"{self.name}:{self.length:.6f}"
        else:
            children_str = ",".join(child.to_newick() for child in self.children)
            if self.parent is None:
                return f"({children_str});"
            else:
                return f"({children_str}):{self.length:.6f}"


# =====================================================================
# Neighbor-Joining algorithm
# =====================================================================


def neighbor_joining(dist_matrix, labels):
    """Neighbor-Joining algorithm implementation"""
    n = len(labels)
    if n < 2:
        return NJTreeNode(labels[0]) if n == 1 else None

    D = dist_matrix.copy().astype(float)
    nodes = {i: NJTreeNode(labels[i]) for i in range(n)}
    active = list(range(n))
    next_id = n

    while len(active) > 2:
        m = len(active)
        row_sums = {i: sum(D[i, j] for j in active if j != i) for i in active}

        min_q = float("inf")
        min_pair = (active[0], active[1])

        for idx_i, i in enumerate(active):
            for j in active[idx_i + 1 :]:
                q = (m - 2) * D[i, j] - row_sums[i] - row_sums[j]
                if q < min_q:
                    min_q = q
                    min_pair = (i, j)

        i, j = min_pair

        if m > 2:
            dist_i = 0.5 * D[i, j] + (row_sums[i] - row_sums[j]) / (2 * (m - 2))
            dist_j = D[i, j] - dist_i
        else:
            dist_i = dist_j = D[i, j] / 2

        dist_i = max(0, dist_i)
        dist_j = max(0, dist_j)

        new_node = NJTreeNode(f"internal_{next_id}")
        nodes[i].length = dist_i
        nodes[j].length = dist_j
        new_node.add_child(nodes[i])
        new_node.add_child(nodes[j])

        new_id = next_id
        next_id += 1

        new_size = max(D.shape[0], new_id + 1)
        if new_size > D.shape[0]:
            new_D = np.zeros((new_size, new_size))
            new_D[: D.shape[0], : D.shape[1]] = D
            D = new_D

        for k in active:
            if k != i and k != j:
                new_dist = max(0, (D[i, k] + D[j, k] - D[i, j]) / 2)
                D[new_id, k] = new_dist
                D[k, new_id] = new_dist

        nodes[new_id] = new_node
        active.remove(i)
        active.remove(j)
        active.append(new_id)

    if len(active) == 2:
        i, j = active
        final_dist = D[i, j] / 2
        root = NJTreeNode("root")
        nodes[i].length = final_dist
        nodes[j].length = final_dist
        root.add_child(nodes[i])
        root.add_child(nodes[j])
        return root
    else:
        return nodes[active[0]]


# =====================================================================
# Color and clustering functions
# =====================================================================


def create_distinct_base_palette(n_colors):
    """Create a visually distinct base color palette"""
    if n_colors <= 0:
        return []

    if n_colors == 7:
        optimized_7_colors = [
            "#E41A1C",
            "#FF7F00",
            "#FFD700",
            "#4DAF4A",
            "#377EB8",
            "#984EA3",
            "#F781BF",
        ]
        return [mcolors.hex2color(c) for c in optimized_7_colors]

    handpicked_colors = [
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#FF7F00",
        "#FFD700",
        "#F781BF",
        "#A65628",
        "#00CED1",
        "#8B4513",
        "#00FF7F",
        "#DC143C",
    ]

    if n_colors > len(handpicked_colors):
        tab20 = sns.color_palette("tab20", 20)
        tab20_hex = [mcolors.rgb2hex(c) for c in tab20]
        all_colors = handpicked_colors + [
            c for c in tab20_hex if c not in handpicked_colors
        ]
    else:
        all_colors = handpicked_colors

    return [mcolors.hex2color(c) for c in all_colors[:n_colors]]


def get_subcluster_colors(base_color, n_subclusters):
    """Generate gradient subcluster colors from a base color"""
    r, g, b = base_color[:3]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    colors = []
    for i in range(n_subclusters):
        if n_subclusters == 1:
            new_s, new_v = s, v
        else:
            ratio = i / (n_subclusters - 1)
            new_s = 0.4 + ratio * 0.6
            new_v = 0.95 - ratio * 0.35

        new_r, new_g, new_b = colorsys.hsv_to_rgb(h, new_s, new_v)
        colors.append((new_r, new_g, new_b))

    return colors


def get_subtree_leaves(node):
    """Get all leaf node names under a given node"""
    if node.is_tip():
        return [node.name]
    leaves = []
    for child in node.children:
        leaves.extend(get_subtree_leaves(child))
    return leaves


def depth_based_tree_clustering(tree_root, depth_threshold):
    """Cut NJ tree based on depth threshold"""
    clusters = []

    def traverse(node, current_depth):
        if node.is_tip():
            clusters.append([node.name])
            return

        for child in node.children:
            child_depth = current_depth + child.length
            if child_depth >= depth_threshold:
                leaves = get_subtree_leaves(child)
                clusters.append(leaves)
            else:
                traverse(child, child_depth)

    traverse(tree_root, 0)

    cluster_labels = {}
    for cluster_idx, leaves in enumerate(clusters):
        for leaf in leaves:
            cluster_labels[leaf] = cluster_idx

    return cluster_labels, len(clusters)


def find_depth_for_n_clusters(tree_root, target_n_clusters, max_depth):
    """Binary search for depth threshold yielding target cluster count"""
    low, high = 0.0, max_depth
    best_depth = max_depth / 2
    best_diff = float("inf")

    for _ in range(50):
        mid = (low + high) / 2
        labels, n = depth_based_tree_clustering(tree_root, mid)

        if n == target_n_clusters:
            return mid, labels, n
        elif n < target_n_clusters:
            low = mid
        else:
            high = mid

        if abs(n - target_n_clusters) < best_diff:
            best_diff = abs(n - target_n_clusters)
            best_depth = mid

    labels, n = depth_based_tree_clustering(tree_root, best_depth)
    return best_depth, labels, n


def get_max_tree_depth(tree_root):
    """Get maximum depth of the tree"""
    tips = list(tree_root.tips())

    def get_node_depth(node):
        depth = 0
        current = node
        while current.parent is not None:
            depth += current.length
            current = current.parent
        return depth

    return max(get_node_depth(tip) for tip in tips)


def get_hierarchical_clusters(
    tree_root, major_depth_threshold, minor_depth_threshold, max_subclusters=3
):
    """Get two-level hierarchical clustering"""
    major_labels, n_major = depth_based_tree_clustering(
        tree_root, major_depth_threshold
    )
    minor_labels, n_minor = depth_based_tree_clustering(
        tree_root, minor_depth_threshold
    )

    hierarchical_labels = {}
    for lang in major_labels:
        hierarchical_labels[lang] = (major_labels[lang], minor_labels[lang])

    major_to_minors = {}
    for lang, (major_id, minor_id) in hierarchical_labels.items():
        if major_id not in major_to_minors:
            major_to_minors[major_id] = set()
        major_to_minors[major_id].add(minor_id)

    major_minor_mapping = {}
    for major_id, minor_ids in major_to_minors.items():
        sorted_minors = sorted(minor_ids)
        n_minors = len(sorted_minors)

        if n_minors <= max_subclusters:
            major_minor_mapping[major_id] = {
                old: new for new, old in enumerate(sorted_minors)
            }
        else:
            major_minor_mapping[major_id] = {}
            for new_id, old_id in enumerate(sorted_minors):
                mapped_id = min(
                    new_id * max_subclusters // n_minors, max_subclusters - 1
                )
                major_minor_mapping[major_id][old_id] = mapped_id

    final_labels = {}
    for lang, (major_id, minor_id) in hierarchical_labels.items():
        final_labels[lang] = (major_id, major_minor_mapping[major_id][minor_id])

    subcluster_counts = {}
    for major_id in major_to_minors:
        subcluster_counts[major_id] = len(set(major_minor_mapping[major_id].values()))

    return final_labels, n_major, subcluster_counts


# =====================================================================
# Plot function 1: Hierarchical clustering dendrogram
# =====================================================================


def plot_rectangular_phylogram_hierarchical(
    tree_root,
    hierarchical_labels,
    n_major_clusters,
    subcluster_counts,
    save_path,
    lang_full_names=None,
    lang_family_map=None,
    lang_to_color=None,
    major_palette=None,
    figsize=(20, 24),
    label_fontsize=14,
):
    """Plot rectangular phylogram with hierarchical clustering"""
    tips = list(tree_root.tips())
    n_tips = len(tips)
    tip_y = {tip.name: i for i, tip in enumerate(tips)}

    def get_node_depth(node):
        depth = 0
        current = node
        while current.parent is not None:
            depth += current.length
            current = current.parent
        return depth

    max_depth = max(get_node_depth(tip) for tip in tips)
    if max_depth == 0:
        max_depth = 1

    coords = {}
    edges = []

    def process_node(node, x_scale=10):
        if node.is_tip():
            x = get_node_depth(node) / max_depth * x_scale
            y = tip_y[node.name]
            coords[id(node)] = (x, y, node.name)
            return y
        else:
            child_ys = []
            for child in node.children:
                child_y = process_node(child, x_scale)
                child_ys.append(child_y)

            y = np.mean(child_ys)
            x = get_node_depth(node) / max_depth * x_scale
            coords[id(node)] = (x, y, None)

            for child in node.children:
                child_x, child_y, _ = coords[id(child)]
                edges.append(((x, y), (x, child_y), "vertical"))
                edges.append(((x, child_y), (child_x, child_y), "horizontal"))

            return y

    process_node(tree_root)

    # Color assignment: prefer passed-in mapping to stay consistent with geo map
    if lang_to_color is None or major_palette is None:
        # Generate internally if not provided (backward compatible)
        major_palette = create_distinct_base_palette(n_major_clusters)
        if n_major_clusters >= 3:
            major_palette[1], major_palette[2] = major_palette[2], major_palette[1]

        lang_to_color = {}
        subcluster_colors_cache = {}

        for lang, (major_id, minor_id) in hierarchical_labels.items():
            base_color = major_palette[major_id]
            n_subs = subcluster_counts[major_id]

            if (major_id, n_subs) not in subcluster_colors_cache:
                subcluster_colors_cache[(major_id, n_subs)] = get_subcluster_colors(
                    base_color, n_subs
                )

            lang_to_color[lang] = subcluster_colors_cache[(major_id, n_subs)][minor_id]

    fig, ax = plt.subplots(figsize=figsize)

    for (x1, y1), (x2, y2), edge_type in edges:
        ax.plot([x1, x2], [y1, y2], "k-", linewidth=0.8, alpha=0.7)

    for tip in tips:
        x, y, name = coords[id(tip)]
        lang = name
        major_id, minor_id = hierarchical_labels.get(lang, (0, 0))
        color = lang_to_color.get(lang, "gray")

        ax.scatter(
            [x], [y], s=80, color=color, zorder=3, edgecolor="black", linewidth=0.5
        )

        # Label: full name (cluster, code, family)
        full_name = lang_full_names.get(lang, lang) if lang_full_names else lang
        family = lang_family_map.get(lang, "?") if lang_family_map else "?"
        label_text = f"{full_name} (C{major_id+1}.{minor_id+1}, {lang}, {family})"
        ax.text(
            x + 0.2,
            y,
            label_text,
            fontsize=label_fontsize,
            fontweight="bold",
            color=color,
            va="center",
            ha="left",
        )

    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-1, n_tips)
    ax.set_xlabel("Distance", fontsize=16, fontweight="bold")
    ax.tick_params(axis='x', labelsize=15)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    ax.set_title(
        f"NJ Tree with Hierarchical Clustering\n{n_major_clusters} major clusters",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=major_palette[i],
            label=f"Cluster {i+1} ({subcluster_counts.get(i, 1)} sub)",
        )
        for i in range(n_major_clusters)
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=18,
        ncol=1,
        title="Major Clusters",
        title_fontsize=19,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Hierarchical tree saved → {save_path}")


# =====================================================================
# Plot function 2: Geographic scatter map
# =====================================================================

# Geographic coordinates
language_coords = {
    "zh": (116.4, 39.9),
    "ko": (126.97, 37.56),
    "ja": (139.69, 35.68),
    "vi": (105.85, 21.03),
    "fr": (2.35, 48.85),
    "gl": (-8.72, 42.23),
    "it": (12.5, 41.9),
    "es": (-3.7, 40.4),
    "ast": (-5.85, 43.36),
    "ca": (2.17, 41.38),
    "oc": (1.44, 43.6),
    "pt": (-9.14, 38.72),
    "ro": (26.1, 44.43),
    "nl": (4.9, 52.37),
    "de": (13.4, 52.52),
    "af": (18.42, -33.93),
    "da": (12.57, 55.68),
    "no": (10.75, 59.91),
    "sv": (18.06, 59.33),
    "fi": (24.94, 60.17),
    "hu": (19.04, 47.5),
    "et": (24.75, 59.44),
    "lv": (24.11, 56.95),
    "lt": (25.28, 54.68),
    "sq": (19.82, 41.33),
    "el": (23.73, 37.98),
    "mk": (21.43, 41.99),
    "uk": (30.52, 50.45),
    "ru": (37.62, 55.75),
    "cs": (14.42, 50.08),
    "pl": (21.01, 52.23),
    "bs": (18.41, 43.85),
    "bg": (23.32, 42.7),
    "hr": (15.98, 45.81),
    "sr": (20.47, 44.82),
    "sk": (17.11, 48.15),
    "sl": (14.51, 46.05),
    "tr": (28.97, 41.01),
    "hi": (77.2, 28.61),
    "id": (106.85, -6.2),
    "ms": (101.7, 3.14),
    "he": (35.21, 31.77),
    "lb": (6.13, 49.61),
    "fy": (5.8, 53.2),
    "yi": (34.8, 31.9),
    "is": (-21.82, 64.13),
    "hy": (44.51, 40.18),
    "ka": (44.79, 41.71),
    "be": (27.57, 53.9),
    "br": (-3.13, 48.4),
    "ga": (-6.26, 53.34),
    "gd": (-4.25, 55.86),
    "cy": (-3.18, 51.48),
    "az": (49.89, 40.39),
    "uz": (69.28, 41.31),
    "bn": (88.36, 22.57),
    "ta": (80.27, 13.08),
    "my": (96.16, 16.85),
    "lo": (102.6, 17.97),
    "th": (100.5, 13.75),
    "fa": (51.42, 35.68),
    "ar": (39.2, 21.5),
    "ps": (69.17, 34.53),
    "so": (45.33, 2.05),
    "sw": (36.82, -1.29),
    "en": (-0.13, 51.51),
    "ba": (56.04, 54.73),
    "kk": (71.43, 51.13),
    "gu": (72.57, 23.03),
    "kn": (77.59, 12.97),
    "mr": (72.88, 19.08),
    "ne": (85.32, 27.71),
    "or": (85.84, 20.27),
    "pa": (75.85, 30.9),
    "sd": (68.36, 25.38),
    "si": (79.86, 6.93),
    "ur": (67.01, 24.86),
    "ceb": (123.89, 10.31),
    "ilo": (120.59, 16.4),
    "jv": (112.75, -7.26),
    "mg": (47.52, -18.91),
    "ml": (76.27, 9.93),
    "su": (107.62, -6.92),
    "tl": (121.77, 13.41),
    "km": (104.92, 11.56),
    "mn": (106.91, 47.92),
    "am": (38.74, 9.03),
    "ff": (-15.98, 14.69),
    "ha": (8.52, 11.85),
    "ig": (7.03, 5.5),
    "ln": (15.27, -4.32),
    "lg": (32.58, 0.31),
    "ns": (28.23, -25.75),
    "ss": (31.13, -26.32),
    "tn": (25.91, -24.65),
    "wo": (-17.45, 14.69),
    "xh": (27.91, -32.3),
    "yo": (3.38, 6.52),
    "zu": (31.02, -29.85),
    "ht": (-72.33, 18.54),
}


def plot_geo_points_map_fullname(
    lang_color_map,
    title,
    save_path,
    selected_langs,
    lang_full_names,
    lang_family_map,
    extent=None,
    fontsize=6,
):
    """Plot language scatter map with full-name labels (uses adjustText to avoid overlap)"""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Warning: cartopy not installed. Skipping geo map.")
        return

    # Try importing adjustText
    try:
        from adjustText import adjust_text

        HAS_ADJUST_TEXT = True
    except ImportError:
        HAS_ADJUST_TEXT = False
        print(
            "Note: adjustText not installed. Labels may overlap. Install with: pip install adjustText"
        )

    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())

    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.set_title(title, fontsize=14)

    texts = []
    x_coords = []
    y_coords = []

    for lang in selected_langs:
        if lang in language_coords:
            lon, lat = language_coords[lang]

            # Filter out languages outside extent
            if extent is not None:
                lon_min, lon_max, lat_min, lat_max = extent
                if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                    continue

            color = lang_color_map.get(lang, "gray")

            # Full name (family) format
            full_name = lang_full_names.get(lang, lang)
            family = lang_family_map.get(lang, "?")
            label = f"{full_name} ({family})"

            ax.plot(
                lon,
                lat,
                "o",
                transform=ccrs.PlateCarree(),
                color=color,
                markersize=5,
                markeredgecolor="black",
                markeredgewidth=0.3,
                zorder=10,
            )

            # Collect text objects for later adjustment (bold)
            txt = ax.text(
                lon,
                lat,
                label,
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold",
                transform=ccrs.PlateCarree(),
                zorder=20,
            )
            texts.append(txt)
            x_coords.append(lon)
            y_coords.append(lat)

    # Use adjustText to auto-adjust label positions to avoid overlap
    if HAS_ADJUST_TEXT and len(texts) > 0:
        adjust_text(
            texts,
            x=x_coords,
            y=y_coords,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.3, alpha=0.5),
            expand_points=(1.5, 1.5),
            force_points=(0.5, 0.5),
            force_text=(0.3, 0.3),
            only_move={"points": "y", "texts": "xy"},
        )

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Geo map saved → {save_path}")


# Language family color mapping (optimized for distinctness)
# Strategy: one hue per family, use light/dark variants to differentiate
FAMILY_COLORS = {
    "IE": "#E41A1C",  # bright red - Indo-European
    "UR": "#377EB8",  # medium blue - Uralic
    "TK": "#B8860B",  # dark gold - Turkic (vs JP bright yellow)
    "AA": "#9932CC",  # orchid purple - Afro-Asiatic (vs TAI dark purple)
    "ST": "#FF7F00",  # orange - Sino-Tibetan
    "JP": "#FFEC00",  # bright yellow - Japonic
    "KR": "#8B4513",  # dark brown - Koreanic
    "AN": "#FF69B4",  # hot pink - Austronesian
    "AS": "#20B2AA",  # light sea green - Austroasiatic
    "TAI": "#4B0082",  # indigo - Tai-Kadai (vs AA light purple)
    "DR": "#00BFFF",  # deep sky blue - Dravidian (vs UR, MO, AS)
    "KA": "#32CD32",  # lime green - Kartvelian
    "MO": "#000080",  # navy blue - Mongolic (vs UR medium blue)
    "NC": "#006400",  # dark green - Niger-Congo (vs KA lime green)
}

FAMILY_FULL_NAMES = {
    "IE": "Indo-European",
    "UR": "Uralic",
    "TK": "Turkic",
    "AA": "Afro-Asiatic",
    "ST": "Sino-Tibetan",
    "JP": "Japonic",
    "KR": "Koreanic",
    "AN": "Austronesian",
    "AS": "Austroasiatic",
    "TAI": "Tai-Kadai",
    "DR": "Dravidian",
    "KA": "Kartvelian",
    "MO": "Mongolic",
    "NC": "Niger-Congo",
}


def plot_geo_map_by_family(
    title,
    save_path,
    selected_langs,
    lang_full_names,
    lang_family_map,
    lang_cluster_color_map,
    extent=None,
    fontsize=6,
    legend_loc="upper right",
):
    """Plot map with family-colored text (dots use cluster colors, text colored by family)"""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Warning: cartopy not installed. Skipping geo map.")
        return

    # Try importing adjustText
    try:
        from adjustText import adjust_text

        HAS_ADJUST_TEXT = True
    except ImportError:
        HAS_ADJUST_TEXT = False
        print("Note: adjustText not installed. Labels may overlap.")

    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())

    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.set_title(title, fontsize=14)

    texts = []
    x_coords = []
    y_coords = []

    # Collect observed language families
    families_used = set()

    for lang in selected_langs:
        if lang in language_coords:
            lon, lat = language_coords[lang]

            # Filter out languages outside extent
            if extent is not None:
                lon_min, lon_max, lat_min, lat_max = extent
                if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                    continue

            family = lang_family_map.get(lang, "?")
            families_used.add(family)

            # Dots use cluster colors (consistent with fullname map)
            point_color = lang_cluster_color_map.get(lang, "gray")

            # Text uses family color
            text_color = FAMILY_COLORS.get(family, "gray")

            # Show full language name only
            full_name = lang_full_names.get(lang, lang)

            # Dot style consistent with fullname map
            ax.plot(
                lon,
                lat,
                "o",
                transform=ccrs.PlateCarree(),
                color=point_color,
                markersize=5,
                markeredgecolor="black",
                markeredgewidth=0.3,
                zorder=10,
            )

            # Text in family color, bold
            txt = ax.text(
                lon,
                lat,
                full_name,
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold",
                color=text_color,
                transform=ccrs.PlateCarree(),
                zorder=20,
            )
            texts.append(txt)
            x_coords.append(lon)
            y_coords.append(lat)

    # Use adjustText to auto-adjust label positions to avoid overlap
    if HAS_ADJUST_TEXT and len(texts) > 0:
        adjust_text(
            texts,
            x=x_coords,
            y=y_coords,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.3, alpha=0.5),
            expand_points=(1.5, 1.5),
            force_points=(0.5, 0.5),
            force_text=(0.3, 0.3),
            only_move={"points": "y", "texts": "xy"},
        )

    # Add legend (upper right)
    from matplotlib.patches import Patch

    legend_elements = []
    for family in sorted(families_used):
        if family in FAMILY_COLORS:
            color = FAMILY_COLORS[family]
            full_name = FAMILY_FULL_NAMES.get(family, family)
            legend_elements.append(
                Patch(
                    facecolor=color, edgecolor="black", linewidth=0.5, label=full_name
                )
            )

    ax.legend(
        handles=legend_elements,
        loc=legend_loc,
        fontsize=8,
        title="Language Family",
        title_fontsize=9,
        framealpha=0.9,
    )

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Family-colored geo map saved → {save_path}")


# =====================================================================
# Heatmap function (ordered by NJ clustering)
# =====================================================================


def get_nj_tree_leaf_order(tree_root):
    """Get NJ tree leaf order (depth-first traversal)"""
    order = []

    def traverse(node):
        if node.is_tip():
            order.append(node.name)
        else:
            for child in node.children:
                traverse(child)

    traverse(tree_root)
    return order


def plot_heatmap_with_clusters(
    dist_matrix,
    lang_list,
    tree_order,
    lang_to_fine_color,
    n_major_clusters,
    save_path,
):
    """Plot heatmap ordered by NJ clustering (labels colored by cluster)"""

    # Reorder distance matrix by tree order (reversed: left-to-right maps to NJ tree top-to-bottom)
    reversed_tree_order = list(reversed(tree_order))
    order_indices = [
        lang_list.index(lang) for lang in reversed_tree_order if lang in lang_list
    ]
    ordered_langs = [lang_list[i] for i in order_indices]

    # Reorder matrix
    reordered_matrix = dist_matrix[np.ix_(order_indices, order_indices)]
    d = len(ordered_langs)

    # Draw heatmap
    fig, ax = plt.subplots(figsize=(14, 14))
    im = ax.imshow(reordered_matrix, cmap="viridis", aspect="equal")

    # Add grid lines (gaps between cells)
    ax.set_xticks(np.arange(d + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(d + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

    # Set tick positions (major ticks for labels)
    ax.set_xticks(np.arange(d))
    ax.set_yticks(np.arange(d))

    # Remove tick marks
    ax.tick_params(axis="both", which="both", length=0)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Move X axis to top
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Clear default labels (we add colored labels manually)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Manually add colored labels
    # Top labels (horizontal, shifted up)
    for i, lang in enumerate(ordered_langs):
        color = lang_to_fine_color.get(lang, "black")
        ax.text(
            i,
            -1.2,
            lang,
            ha="center",
            va="top",
            fontsize=7,
            fontweight="bold",
            color=color,
            rotation=0,
        )

    # Left-side labels
    for i, lang in enumerate(ordered_langs):
        color = lang_to_fine_color.get(lang, "black")
        ax.text(
            -0.7,
            i,
            lang,
            ha="right",
            va="center",
            fontsize=7,
            fontweight="bold",
            color=color,
        )

    # Add value annotations
    for i in range(d):
        for j in range(d):
            ax.text(
                j,
                i,
                f"{reordered_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=4,
                fontweight="bold",
            )

    # Set axis limits
    ax.set_xlim(-0.5, d - 0.5)
    ax.set_ylim(d - 0.5, -0.5)

    # Hide spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"Language Distance Matrix\n({n_major_clusters} clusters, NJ order)",
        fontsize=14,
        pad=40,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Heatmap saved → {save_path}")


# =====================================================================
# IE tree function (colored by NJ clustering)
# =====================================================================

# IE tree structure
IE_TREE_ORIGINAL = (
    "Indo-European",
    [
        (
            "Romance",
            [
                (
                    "Western-Romance",
                    [
                        ("Gallo-Romance", ["fr", "oc"]),
                        ("Iberian-Romance", ["pt", "gl", "es", "ast", "ca"]),
                        ("Italo-Dalmatian", ["it"]),
                    ],
                ),
                ("Eastern-Romance", ["ro"]),
            ],
        ),
        (
            "Germanic",
            [
                ("West-Germanic", ["de", "nl", "af", "fy", "lb", "yi", "en"]),
                ("North-Germanic", ["da", "sv", "no", "is"]),
            ],
        ),
        (
            "Celtic",
            [
                ("Brythonic", ["cy", "br"]),
                ("Goidelic", ["ga", "gd"]),
            ],
        ),
        (
            "Balto-Slavic",
            [
                ("Baltic", ["lv", "lt"]),
                (
                    "Slavic",
                    [
                        ("East-Slavic", ["ru", "uk", "be"]),
                        ("West-Slavic", ["pl", "cs", "sk"]),
                        ("South-Slavic", ["bg", "mk", "sr", "hr", "sl", "bs"]),
                    ],
                ),
            ],
        ),
        ("Hellenic", ["el"]),
        ("Albanian", ["sq"]),
        ("Armenian", ["hy"]),
        (
            "Indo-Iranian",
            [
                (
                    "Indo-Aryan",
                    ["hi", "bn", "ur", "si", "mr", "pa", "ne", "gu", "or", "sd"],
                ),
                ("Iranian", ["fa", "ps"]),
            ],
        ),
    ],
)

# Tree plot parameters (horizontal layout: root left, leaves right)
TREE_YSTEP = 1.2  # Y spacing between leaf nodes
TREE_XSTEP = 1.5  # X spacing per depth level
TREE_FIGSIZE = (14, 20)  # width x height


def prune_ie_tree(node, keep_langs):
    """Prune IE tree to only keep languages in keep_langs.

    Removes leaf languages not in keep_langs, and removes internal nodes
    that become empty after pruning. Collapses single-child internal nodes.
    """
    label, children = node
    new_children = []
    for c in children:
        if isinstance(c, str):
            if c in keep_langs:
                new_children.append(c)
        else:
            pruned = prune_ie_tree(c, keep_langs)
            if pruned is not None:
                new_children.append(pruned)
    if not new_children:
        return None
    return (label, new_children)


def collect_leaves(node):
    """Return all leaf language codes in the tree"""
    label, children = node
    leaves = set()
    for c in children:
        if isinstance(c, str):
            leaves.add(c)
        else:
            leaves |= collect_leaves(c)
    return leaves


def compute_branch_colors(node, cluster_color_map, present_langs):
    """Compute average color for each branch node (based on descendant languages)"""

    def get_present_leaves(n):
        label, children = n
        leaves = set()
        for c in children:
            if isinstance(c, str):
                if c in present_langs:
                    leaves.add(c)
            else:
                leaves |= get_present_leaves(c)
        return leaves

    def compute_colors_recursive(n):
        label, children = n
        colors = {}

        for c in children:
            if not isinstance(c, str):
                colors.update(compute_colors_recursive(c))

        present_leaves = get_present_leaves(n)
        if present_leaves:
            rgb_colors = [
                cluster_color_map.get(lang, (0.5, 0.5, 0.5)) for lang in present_leaves
            ]
            avg_rgb = tuple(np.mean(rgb_colors, axis=0))
            colors[label] = avg_rgb
        else:
            colors[label] = (0.5, 0.5, 0.5)

        return colors

    return compute_colors_recursive(node)


def layout_ie_tree_horizontal(node, xstep=2.5, ystep=1.2):
    """
    Compute horizontal layout positions for IE tree (root left, leaves right).

    Returns:
        positions: dict, node name -> (x, y) coordinates
        segments: list, line segments [(x1, y1, x2, y2), ...]
    """
    positions = {}
    segments = []

    # Step 1: compute depth and leaf count for each node
    def count_leaves(n):
        """Return number of leaves under a node"""
        label, children = n
        total = 0
        for c in children:
            if isinstance(c, str):
                total += 1
            else:
                total += count_leaves(c)
        return total

    # Step 2: recursive layout
    def _layout(n, depth, y_cursor):
        """
        Returns: (y_cursor_after, y_center)
        """
        label, children = n
        child_y_centers = []

        for child in children:
            if isinstance(child, str):
                # Leaf node: place at rightmost position
                x = (depth + 1) * xstep
                y = y_cursor
                positions[child] = (x, y)
                child_y_centers.append(y)
                y_cursor += ystep
            else:
                # Subtree: recurse
                y_cursor, child_y_center = _layout(child, depth + 1, y_cursor)
                child_y_centers.append(child_y_center)

        # Current node Y = center of children
        y_center = (
            sum(child_y_centers) / len(child_y_centers) if child_y_centers else y_cursor
        )
        x_center = depth * xstep
        positions[label] = (x_center, y_center)

        # Draw orthogonal connector lines
        for child in children:
            if isinstance(child, str):
                cx, cy = positions[child]
            else:
                cx, cy = positions[child[0]]
            # Lines from parent to child x position
            segments.append((x_center, y_center, x_center, cy))  # vertical
            segments.append((x_center, cy, cx, cy))  # horizontal

        return y_cursor, y_center

    _layout(node, 0, 0)
    return positions, segments


def plot_ie_tree_with_clusters(
    tree_structure,
    lang_color_map,
    save_path,
    lang_full_names=None,
):
    """Plot horizontal IE tree colored by clusters (root left, leaves right, NJ tree style)"""
    our_language_codes = set(lang_color_map.keys())

    # Prune tree: keep present languages + English for reference
    ie_codes_all = collect_leaves(tree_structure)
    keep_langs = (ie_codes_all & our_language_codes) | {"en"}
    pruned_tree = prune_ie_tree(tree_structure, keep_langs)
    if pruned_tree is None:
        print("Warning: no IE languages present, skipping IE tree.")
        return

    # Horizontal layout (on pruned tree)
    positions, segments = layout_ie_tree_horizontal(
        pruned_tree, xstep=TREE_XSTEP, ystep=TREE_YSTEP
    )

    ie_codes = collect_leaves(pruned_tree)
    present = sorted(ie_codes & our_language_codes)
    present_set = set(present)

    # Compute branch node colors (average of descendant cluster colors)
    branch_colors = compute_branch_colors(pruned_tree, lang_color_map, present_set)

    plt.figure(figsize=TREE_FIGSIZE)

    # Draw connecting lines
    for x1, y1, x2, y2 in segments:
        plt.plot([x1, x2], [y1, y2], linewidth=1.2, color="#888888")

    # Internal node labels (dynamically collect from pruned tree)
    internal_labels = set(positions.keys()) - ie_codes
    # Also ensure we have the standard ones
    internal_labels |= {
        "Indo-European",
        "Romance",
        "Western-Romance",
        "Gallo-Romance",
        "Iberian-Romance",
        "Italo-Dalmatian",
        "Eastern-Romance",
        "Germanic",
        "Celtic",
        "Brythonic",
        "Goidelic",
        "Balto-Slavic",
        "Slavic",
        "Baltic",
        "Hellenic",
        "Albanian",
        "Indo-Iranian",
        "Indo-Aryan",
        "Iranian",
        "North-Germanic",
        "West-Germanic",
        "East-Slavic",
        "West-Slavic",
        "South-Slavic",
        "Armenian",
    }

    for label, (x, y) in positions.items():
        if label in internal_labels:
            # Use black for branch labels
            text_color = "black"

            # Root node: larger font, placed above
            if label == "Indo-European":
                fontsize = 22
                plt.text(
                    x,
                    y + 0.35,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight="bold",
                    color=text_color,
                )
            else:
                fontsize = 16
                # Non-root: label at midpoint of horizontal line above (x - half step)
                label_x = x - TREE_XSTEP * 0.5
                plt.text(
                    label_x,
                    y + 0.25,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight="bold",
                    color=text_color,
                )

    # Leaf nodes
    for lab in ie_codes:
        if lab not in positions:
            continue
        x, y = positions[lab]

        # Get full language name
        if lang_full_names:
            display_name = lang_full_names.get(lab, lab)
        else:
            display_name = lab

        if lab in present:
            color = lang_color_map.get(lab, "#1f77b4")
            plt.scatter(
                [x],
                [y],
                s=80,
                color=color,
                marker="o",
                zorder=3,
                edgecolor="black",
                linewidth=0.5,
            )
            # Leaf label on right side, full name, bold
            plt.text(
                x + 0.15,
                y,
                display_name,
                ha="left",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=color,
            )
        else:
            # Languages not in dataset shown as gray triangles
            plt.scatter([x], [y], s=60, color="#B0B0B0", marker="^", zorder=3)
            plt.text(
                x + 0.15,
                y,
                display_name,
                ha="left",
                va="center",
                fontsize=15,
                fontweight="bold",
                color="#B0B0B0",
            )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ IE tree saved → {save_path}")


# =====================================================================
# Main program
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="m2m", choices=["m2m", "llama3"],
                        help="Model to visualize (m2m or llama3)")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    MODEL = args.model
    TOP_K = cfg["top_k"]
    THR = cfg["threshold"]
    RESULTS_DIR = cfg["results_dir"]
    SUFFIX = cfg["suffix"]
    OUTDIR = f"viz_out_v12_simple/{MODEL}_{TOP_K}_{THR}"
    os.makedirs(OUTDIR, exist_ok=True)
    SUFFIX_CLEAN = SUFFIX.lstrip("_") if SUFFIX else ""

    # Override Haitian Creole family label per model
    language_family_map["ht"] = cfg["ht_family"]

    print("=" * 60)
    print(f"Loading data for {MODEL}")
    print("=" * 60)

    with open(f"{RESULTS_DIR}/selected_{MODEL}_{TOP_K}_{THR}.pkl", "rb") as f:
        sel = pickle.load(f)
    with open(f"{RESULTS_DIR}/all_{MODEL}_distance_dict{SUFFIX}.pkl", "rb") as f:
        all_dist = pickle.load(f)

    all_lang_dict, all_lang_list = get_all_lang()
    idx_sel = sel["selected_indices"]
    lang_sel = sel["selected_languages"]
    lang_idx = [all_lang_list.index(l) for l in lang_sel]

    print(f"Number of selected sentences: {len(idx_sel)}")
    print(f"Number of languages: {len(lang_sel)}")

    # Check for missing indices
    missing_indices = [i for i in idx_sel if i not in all_dist]
    if missing_indices:
        print(f"Missing indices: {len(missing_indices)}")
        idx_sel = [i for i in idx_sel if i in all_dist]

    # Build distance matrix (mean method only)
    sub = [all_dist[i][lang_idx][:, lang_idx] for i in idx_sel]
    tensor = np.stack(sub, axis=0)
    dist_matrix = np.mean(tensor, axis=(0, 3))

    print(f"Distance matrix shape: {dist_matrix.shape}")

    # Build NJ tree
    print("\n" + "=" * 60)
    print("Building Neighbor-Joining tree (mean method)")
    print("=" * 60)

    nj_tree = neighbor_joining(dist_matrix, lang_sel)
    print(f"NJ tree built successfully!")
    print(f"Number of tips: {len(list(nj_tree.tips()))}")

    # Compute cophenetic correlation
    print_cophenetic_correlation(nj_tree, dist_matrix, lang_sel, method_name="mean")

    max_depth = get_max_tree_depth(nj_tree)
    print(f"Max tree depth: {max_depth:.4f}")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations")
    print("=" * 60)

    # Save color maps per cluster count for geo_by_family plots
    saved_color_maps = {}

    for n_major_clusters in N_CLUSTERS_LIST:
        print(f"\n--- {n_major_clusters} major clusters ---")

        # Find depth threshold
        major_depth, _, n_major = find_depth_for_n_clusters(
            nj_tree, n_major_clusters, max_depth
        )
        minor_depth, _, _ = find_depth_for_n_clusters(
            nj_tree, n_major_clusters * MAX_SUBCLUSTERS, max_depth
        )

        # Get hierarchical clustering labels
        hier_labels, n_major, subcluster_counts = get_hierarchical_clusters(
            nj_tree, major_depth, minor_depth, max_subclusters=MAX_SUBCLUSTERS
        )

        print(f"  Major clusters: {n_major}")
        print(f"  Subclusters per major: {subcluster_counts}")

        # Create color mapping (shared by tree and geo plots for consistency)
        major_palette = create_distinct_base_palette(n_major)
        if n_major >= 3:
            major_palette[1], major_palette[2] = major_palette[2], major_palette[1]

        lang_to_major_color = {}
        lang_to_fine_color = {}
        subcluster_colors_cache = {}
        for lang, (major_id, minor_id) in hier_labels.items():
            base_color = major_palette[major_id]
            lang_to_major_color[lang] = base_color
            n_subs = subcluster_counts[major_id]
            if (major_id, n_subs) not in subcluster_colors_cache:
                subcluster_colors_cache[(major_id, n_subs)] = get_subcluster_colors(
                    base_color, n_subs
                )
            lang_to_fine_color[lang] = subcluster_colors_cache[(major_id, n_subs)][
                minor_id
            ]

        # Save color mapping for geo_by_family
        saved_color_maps[n_major] = lang_to_fine_color.copy()

        # 1. Plot hierarchical clustering dendrogram (pre-computed color mapping)
        filename = f"nj_hierarchical_{n_major}clusters_mean_{MODEL}_{TOP_K}_{THR}"
        if SUFFIX_CLEAN:
            filename += f"_{SUFFIX_CLEAN}"
        filename += ".png"
        save_path = os.path.join(OUTDIR, filename)

        plot_rectangular_phylogram_hierarchical(
            nj_tree,
            hier_labels,
            n_major,
            subcluster_counts,
            save_path,
            lang_full_names=all_lang_dict,
            lang_family_map=language_family_map,
            lang_to_color=lang_to_fine_color,
            major_palette=major_palette,
            figsize=cfg["tree_figsize"],
            label_fontsize=cfg["tree_fontsize"],
        )

        # 2. Plot fine-grained geo scatter (with full-name labels, same color mapping)

        # Eurasia version
        filename = f"nj_hier_geo_points_fine_fullname_{n_major}clusters_mean_{MODEL}_{TOP_K}_{THR}"
        if SUFFIX_CLEAN:
            filename += f"_{SUFFIX_CLEAN}"
        filename += "_eurasia.png"
        save_path = os.path.join(OUTDIR, filename)

        plot_geo_points_map_fullname(
            lang_to_fine_color,
            f"Language Clusters - Fine-grained with Full Names\n({n_major} major clusters, Eurasia & Africa)",
            save_path,
            lang_sel,
            all_lang_dict,
            language_family_map,
            extent=EURASIA_EXTENT,
        )

        # Eurasia Extended version (includes Haiti)
        filename = f"nj_hier_geo_points_fine_fullname_{n_major}clusters_mean_{MODEL}_{TOP_K}_{THR}"
        if SUFFIX_CLEAN:
            filename += f"_{SUFFIX_CLEAN}"
        filename += "_eurasia_extended.png"
        save_path = os.path.join(OUTDIR, filename)

        plot_geo_points_map_fullname(
            lang_to_fine_color,
            f"Language Clusters - Fine-grained with Full Names\n({n_major} major clusters, Extended)",
            save_path,
            lang_sel,
            all_lang_dict,
            language_family_map,
            extent=EURASIA_EXTENDED_EXTENT,
            fontsize=5,
        )

        # Global version
        filename = f"nj_hier_geo_points_fine_fullname_{n_major}clusters_mean_{MODEL}_{TOP_K}_{THR}"
        if SUFFIX_CLEAN:
            filename += f"_{SUFFIX_CLEAN}"
        filename += "_global.png"
        save_path = os.path.join(OUTDIR, filename)

        plot_geo_points_map_fullname(
            lang_to_fine_color,
            f"Language Clusters - Fine-grained with Full Names\n({n_major} major clusters, Global)",
            save_path,
            lang_sel,
            all_lang_dict,
            language_family_map,
            extent=None,
        )

        # 3. Plot heatmap (ordered by NJ tree)
        tree_order = get_nj_tree_leaf_order(nj_tree)
        filename = f"heatmap_{n_major}clusters_mean_{MODEL}_{TOP_K}_{THR}"
        if SUFFIX_CLEAN:
            filename += f"_{SUFFIX_CLEAN}"
        filename += ".png"
        save_path = os.path.join(OUTDIR, filename)

        plot_heatmap_with_clusters(
            dist_matrix,
            lang_sel,
            tree_order,
            lang_to_fine_color,
            n_major,
            save_path,
        )

        # 4. Plot IE tree (colored by clustering)
        filename = f"ie_tree_{n_major}clusters_mean_{MODEL}_{TOP_K}_{THR}"
        if SUFFIX_CLEAN:
            filename += f"_{SUFFIX_CLEAN}"
        filename += ".png"
        save_path = os.path.join(OUTDIR, filename)

        plot_ie_tree_with_clusters(
            IE_TREE_ORIGINAL,
            lang_to_fine_color,
            save_path,
            lang_full_names=all_lang_dict,
        )

    # 5. Plot family-colored maps (one per cluster count, dots use cluster colors)
    print("\n--- Language Family Maps ---")

    for n_clusters, cluster_color_map in saved_color_maps.items():
        print(f"  Generating geo_by_family for {n_clusters} clusters...")

        # Eurasia version
        filename = f"geo_by_family_{n_clusters}clusters_mean_{MODEL}_{TOP_K}_{THR}"
        if SUFFIX_CLEAN:
            filename += f"_{SUFFIX_CLEAN}"
        filename += "_eurasia.png"
        save_path = os.path.join(OUTDIR, filename)

        plot_geo_map_by_family(
            f"Languages Colored by Family ({n_clusters} clusters, Eurasia & Africa)",
            save_path,
            lang_sel,
            all_lang_dict,
            language_family_map,
            cluster_color_map,
            extent=EURASIA_EXTENT,
        )

        # Eurasia Extended version (includes Haiti)
        filename = f"geo_by_family_{n_clusters}clusters_mean_{MODEL}_{TOP_K}_{THR}"
        if SUFFIX_CLEAN:
            filename += f"_{SUFFIX_CLEAN}"
        filename += "_eurasia_extended.png"
        save_path = os.path.join(OUTDIR, filename)

        plot_geo_map_by_family(
            f"Languages Colored by Family ({n_clusters} clusters, Extended)",
            save_path,
            lang_sel,
            all_lang_dict,
            language_family_map,
            cluster_color_map,
            extent=EURASIA_EXTENDED_EXTENT,
            fontsize=5,
            legend_loc="upper left",
        )

        # Global version
        filename = f"geo_by_family_{n_clusters}clusters_mean_{MODEL}_{TOP_K}_{THR}"
        if SUFFIX_CLEAN:
            filename += f"_{SUFFIX_CLEAN}"
        filename += "_global.png"
        save_path = os.path.join(OUTDIR, filename)

        plot_geo_map_by_family(
            f"Languages Colored by Family ({n_clusters} clusters, Global)",
            save_path,
            lang_sel,
            all_lang_dict,
            language_family_map,
            cluster_color_map,
            extent=None,
        )

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Output directory: {OUTDIR}")
    print("=" * 60)
