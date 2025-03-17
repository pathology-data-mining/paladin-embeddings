import pickle
from gensim.models.poincare import PoincareModel, PoincareRelations
from anytree import PreOrderIter


def tree_to_relations(tree):
    relations = []
    for node in PreOrderIter(tree):
        if node.parent is not None:
            relations.append((node.parent.name, node.name))
    return relations


# Load the tree from the pickle file
with open('oncotree/oncotree.pkl', 'rb') as f:
    tree = pickle.load(f)

print(f"Loaded tree with {len(list(PreOrderIter(tree)))} nodes, {len([node for node in PreOrderIter(tree) if node.is_leaf])} leaves, and {len([node for node in PreOrderIter(tree) if node.parent is not None])} edges")

relations = tree_to_relations(tree)

# Save the relations to a file
with open('oncotree/relations.txt', 'w') as f:
    for parent, child in relations:
        f.write(f"{parent}\t{child}\n")

# Load the relations
relations = PoincareRelations('oncotree/relations.txt')

# Train the Poincare model
model = PoincareModel(train_data=relations, size=32, negative=2)
model.train(epochs=50)

print("Training complete")
# Save the embeddings
model.save('oncotree/poincare_embeddings')