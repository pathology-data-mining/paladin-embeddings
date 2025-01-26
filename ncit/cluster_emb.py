import pandas as pd
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load entity embeddings
entity_embeddings = torch.load('ncit/ncit_embeddings2/entity_embeddings.pt',
                               map_location=torch.device('cpu'))

mapping = pd.read_csv('ncit/ncit_embeddings2/entity_to_id_with_oncotree.csv')
ordered_embeddings = entity_embeddings[mapping['index'].values.tolist()]

# Convert embeddings to numpy for t-SNE
embeddings_np = ordered_embeddings.detach().numpy()

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_np)

# Create scatter plot
plt.figure(figsize=(15, 15))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

# Add labels for each point
for i, (x, y) in enumerate(embeddings_2d):
    plt.annotate(mapping['ONCOTREE_CODE'].iloc[i], 
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7)

plt.title('t-SNE visualization of entity embeddings')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')

# Save the plot
plt.savefig('ncit/ncit_embeddings2/tsne_embeddings.png', bbox_inches='tight', dpi=300)
plt.close()

# Also print the coordinates with labels
print("\nCoordinates and labels:")
for i, (x, y) in enumerate(embeddings_2d):
    print(f"{mapping['ONCOTREE_CODE'].iloc[i]}: ({x:.2f}, {y:.2f})")


# Create dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
plt.figure(figsize=(20, 10))

# Compute linkage matrix
linkage_matrix = linkage(embeddings_np, method='ward')

# Create heatmap with clustering
from scipy.cluster.hierarchy import leaves_list
import seaborn as sns

# Get the order of leaves from the linkage matrix
leaves_order = leaves_list(linkage_matrix)

# Reorder the distance matrix according to the clustering
ordered_embeddings_np = embeddings_np[leaves_order]
distance_matrix = np.zeros((len(ordered_embeddings_np), len(ordered_embeddings_np)))
for i in range(len(ordered_embeddings_np)):
    for j in range(len(ordered_embeddings_np)):
        distance_matrix[i,j] = np.linalg.norm(ordered_embeddings_np[i] - ordered_embeddings_np[j])

# Create heatmap
plt.figure(figsize=(30, 30))
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)

sns.heatmap(distance_matrix, 
            xticklabels=mapping['ONCOTREE_CODE'].values[leaves_order],
            yticklabels=mapping['ONCOTREE_CODE'].values[leaves_order],
            cmap='viridis',
            square=True)
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('OncoTree Code')
plt.ylabel('Distance')

# Save dendrogram plot
plt.savefig('ncit/ncit_embeddings2/dendrogram.png', bbox_inches='tight', dpi=300)
plt.close()
