import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import umap
import requests
import json
import os


def get_color_map():
    cache_path = "oncotree/oncotree.json"
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            color_map = json.load(f)
    for k, v in color_map.items():
        if v is None:
            color_map[k] = "black"
    else:
        url = "https://oncotree.info/api/tumorTypes/tree"
        response = requests.get(url)
        oncotree = response.json()
        color_map = get_color_map_recursive({}, oncotree['TISSUE'])
        with open(cache_path, 'w') as f:
            json.dump(color_map, f)
    return color_map


def get_color_map_recursive(color_map, tree):
    """
    color_map: dict of color_map
    tree: dict of tree
    """
    color_map[tree["code"]] = tree["color"]
    if "children" in tree.keys():
        for value in tree["children"].values():
            color_map = get_color_map_recursive(color_map, value)
    return color_map

def get_umap_embeddings(euclidean_embeddings):
    cache_path = "oncotree/umap_emb.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            umap_embeddings = pickle.load(f)
    else:
        umap_model = umap.UMAP(n_components=2, random_state=42)
        umap_model.fit(euclidean_embeddings)
        umap_embeddings = umap_model.transform(euclidean_embeddings)
        with open(cache_path, 'wb') as f:
            pickle.dump(umap_embeddings, f)
    return umap_embeddings



if __name__ == "__main__":
    with open('oncotree/euclidean_embeddings.pkl', 'rb') as f:
        euclidean_embeddings = pickle.load(f)
    euclidean_embeddings = pd.DataFrame(euclidean_embeddings).T

    color_map = get_color_map()
    ordered_colors = [color_map[code] for code in euclidean_embeddings.index]
    
    umap_embeddings = get_umap_embeddings(euclidean_embeddings)

    plt.figure(figsize=(4,3))
    sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], hue=euclidean_embeddings.index, palette=ordered_colors)
    
    # Formatting
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("oncotree/umap.png", dpi=300)
    plt.close()

