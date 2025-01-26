import numpy as np
from scipy.spatial.distance import euclidean
import seaborn as sns
import matplotlib.pyplot as plt
from anytree import PreOrderIter
from gensim.models.poincare import PoincareModel
import pickle
import pandas as pd


def poincare_to_euclidean(embedding):
    norm = np.linalg.norm(embedding)
    # riemannian_metric = 2 / (1 - norm ** 2)
    return embedding / (1 + norm)


# Load the tree from the pickle file
with open('oncotree/oncotree.pkl', 'rb') as f:
    tree = pickle.load(f)

# To load the model later
model = PoincareModel.load('oncotree/poincare_embeddings')

# Define the nodes of interest
node_names = [node.name for node in PreOrderIter(tree)]
n_nodes = len(node_names)

# extract euclidean embeddings and hypebolic embeddings
euclidean_embeddings = {}
hyperbolic_embeddings = {}
for node_name in node_names:
    euclidean_embeddings[node_name] = poincare_to_euclidean(model.kv[node_name])
    hyperbolic_embeddings[node_name] = model.kv[node_name]

with open('oncotree/euclidean_embeddings.pkl', 'wb') as f:
    pickle.dump(euclidean_embeddings, f)

# Convert embeddings to lists for JSON serialization
euclidean_embeddings_list = {k: v.tolist() for k, v in euclidean_embeddings.items()}
# print(euclidean_embeddings_list)
hyperbolic_embeddings_list = {k: v.tolist() for k, v in hyperbolic_embeddings.items()}

# Save as JSON
import json

with open('oncotree/euclidean_embeddings.json', 'w') as f:
    json.dump(euclidean_embeddings_list, f)

with open('oncotree/hyperbolic_embeddings.json', 'w') as f:
    json.dump(hyperbolic_embeddings_list, f)

with open('oncotree/hyperbolic_embeddings.pkl', 'wb') as f:
    pickle.dump(hyperbolic_embeddings, f)

# Calculate the distances
hyperbolic_distances = {}
euclidean_distances = {}
hyperbolic_matrix = np.zeros((n_nodes, n_nodes))
euclidean_matrix = np.zeros((n_nodes, n_nodes))
for i, node1 in enumerate(node_names):
    for j, node2 in enumerate(node_names):
        hyperbolic_distance = model.kv.distance(node1, node2)
        hyperbolic_distances[(node1, node2)] = hyperbolic_distance
        hyperbolic_matrix[i, j] = hyperbolic_distance

        euclidean_distance = euclidean(euclidean_embeddings[node1], euclidean_embeddings[node2])
        euclidean_distances[(node1, node2)] = euclidean_distance
        euclidean_matrix[i, j] = euclidean_distance

# Save the distances
with open('oncotree/hyperbolic_distances.pkl', 'wb') as f:
    pickle.dump(hyperbolic_distances, f)

with open('oncotree/euclidean_distances.pkl', 'wb') as f:
    pickle.dump(euclidean_distances, f)

euclidean_df = pd.DataFrame(euclidean_matrix, index=node_names, columns=node_names)
hyperbolic_df = pd.DataFrame(hyperbolic_matrix, index=node_names, columns=node_names)
euclidean_df.to_csv('oncotree/euclidean_distances.csv')
hyperbolic_df.to_csv('oncotree/hyperbolic_distances.csv')
