import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import umap
import requests
import json
import os
import plotly.express as px
import plotly.io as pio

def set_font_sizes(size=8):
    """
    Set all font sizes in matplotlib to the specified size.
    
    Args:
        size (int): Font size to use for all text elements. Default is 8.
    """
    import matplotlib as mpl
    
    # Set font sizes for various elements
    plt.rc('font', size=size)          # controls default text sizes
    plt.rc('axes', titlesize=size)     # fontsize of the axes title
    plt.rc('axes', labelsize=size)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=size)    # legend fontsize
    plt.rc('figure', titlesize=size)   # fontsize of the figure title



def get_color_map():
    cache_path = "oncotree/oncotree.json"
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            color_map = json.load(f)
    else:
        url = "https://oncotree.info/api/tumorTypes/tree"
        response = requests.get(url)
        oncotree = response.json()
        color_map = get_color_map_recursive({}, oncotree['TISSUE'])
        with open(cache_path, 'w') as f:
            json.dump(color_map, f)
    color_map['TISSUE'] = "black"
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

def get_umap_embeddings(euclidean_embeddings, use_cache=True):
    cache_path = "oncotree/umap_emb.pkl"
    if os.path.exists(cache_path) and use_cache:
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
    set_font_sizes(size=8)
    # with open('oncotree/euclidean_embeddings.pkl', 'rb') as f:
    #     euclidean_embeddings = pickle.load(f)
    # euclidean_embeddings = pd.DataFrame(euclidean_embeddings).T
    with open('oncotree/compressed_embeddings.json', 'r') as f:
        euclidean_embeddings = json.load(f)
    euclidean_embeddings = pd.DataFrame(euclidean_embeddings).T
    print("Using compressed embeddings with shape:", euclidean_embeddings.shape)
    
    color_map = get_color_map()
    ordered_colors = [color_map[code] for code in euclidean_embeddings.index]
    
    # Force recalculation of UMAP with compressed embeddings
    umap_embeddings = get_umap_embeddings(euclidean_embeddings, use_cache=False)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'UMAP1': umap_embeddings[:, 0],
        'UMAP2': umap_embeddings[:, 1],
        'Category': euclidean_embeddings.index.tolist(),
        'Color': ordered_colors
    })
    
    # Create interactive plot with Plotly
    fig = px.scatter(
        plot_df,
        x='UMAP1',
        y='UMAP2',
        color='Category',
        color_discrete_sequence=ordered_colors,
        title='OncoTree UMAP Visualization (Compressed Embeddings)',
        width=800,
        height=800,
        hover_data=['Category']  # Explicitly include Category in hover data
    )
    
    # Update layout for cleaner look
    fig.update_layout(
        showlegend=False,  # Hide the legend since we have hover labels
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate='<b>Category:</b> %{customdata[0]}<extra></extra>'
    )
    
    # Save as HTML file for interactive viewing
    fig.write_html("oncotree/umap_interactive_compressed.html")

