import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy
import numpy as np
# from d3heatmap import d3heatmap


# Load the distance dfs
euclidean_distances = pd.read_csv('oncotree/euclidean_distances.csv', index_col=0)
hyperbolic_distances = pd.read_csv('oncotree/hyperbolic_distances.csv', index_col=0)

row_order = euclidean_distances.index.tolist()

small_matrix = euclidean_distances.loc[row_order[:10], row_order[:10]]

# heatmap = d3heatmap.heatmap(small_matrix)
# plt.show()
# exit()

# load the actual embeddings
with open('oncotree/euclidean_embeddings.pkl', 'rb') as f:
    euclidean_embeddings = pickle.load(f)
euclidean_embeddings = pd.DataFrame(euclidean_embeddings).T
euclidean_embeddings = euclidean_embeddings.loc[row_order]

with open('oncotree/hyperbolic_embeddings.pkl', 'rb') as f:
    poincare_embeddings = pickle.load(f)
poincare_embeddings = pd.DataFrame(poincare_embeddings).T
poincare_embeddings = poincare_embeddings.loc[row_order]


# # convert the redundant n*n square matrix form into a condensed nC2 array
# euclidean_distances = ssd.squareform(euclidean_distances.values) # distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j
# # make hyperbolic_distances symmetric
# hyperbolic_distances = (hyperbolic_distances + hyperbolic_distances.T) / 2
# hyperbolic_distances = ssd.squareform(hyperbolic_distances.values)

# euclidean_linkage = scipy.cluster.hierarchy.linkage(euclidean_distances, method='ward')
# hyperbolic_linkage = scipy.cluster.hierarchy.linkage(hyperbolic_distances, method='ward')

# plot the heatmaps with clustering
plt.figure(figsize=(12, 12))
sns.clustermap(euclidean_distances, row_cluster=True, col_cluster=True, cmap='viridis', method='ward')
# sns.clustermap(euclidean_embeddings, row_linkage=euclidean_linkage, col_cluster=True, cmap='viridis')
plt.title('Euclidean distance')
plt.tight_layout()
plt.savefig('oncotree/embedding_heatmaps_euclidean.png', dpi=300)
plt.close()

plt.figure(figsize=(12, 12))
sns.clustermap(hyperbolic_distances, row_cluster=True, col_cluster=True, cmap='viridis', method='ward')
# sns.clustermap(poincare_embeddings, row_linkage=hyperbolic_linkage, col_cluster=True, cmap='viridis')
plt.title('Hyperbolic distance')
plt.tight_layout()
plt.savefig('oncotree/embedding_heatmap_hyperbolic.png', dpi=300)
plt.close()

# load the distances
with open('oncotree/hyperbolic_distances.pkl', 'rb') as f:
    hyperbolic_distances = pickle.load(f)

with open('oncotree/euclidean_distances.pkl', 'rb') as f:
    euclidean_distances = pickle.load(f)

# Extract distances for plotting
keys = list(hyperbolic_distances.keys())
hyperbolic_distances_list = [hyperbolic_distances[key] for key in keys]
euclidean_distances_list = [euclidean_distances[key] for key in keys]

# Plot the correlation
plt.figure(figsize=(6, 6))
p = sns.scatterplot(x=hyperbolic_distances_list, y=euclidean_distances_list, alpha=0.1)
plt.title('Plot of Hyperbolic vs Euclidean Distances')
plt.xlabel('Hyperbolic Distance')
plt.ylabel('Euclidean Distance')
r2 = np.corrcoef(hyperbolic_distances_list, euclidean_distances_list)[0, 1]
plt.text(4, 0.2, f'$R = {r2:.2f}$', fontsize=12)
plt.tight_layout()
plt.savefig('oncotree/distance_correlation_density.png', dpi=300)
plt.close()

# Print the embeddings for the nodes of interest
# for node in nodes_of_interest:
#     embedding = model.kv[node]
#     print(f"Embedding for '{node}': {embedding}")
