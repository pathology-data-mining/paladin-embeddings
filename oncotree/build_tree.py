import requests
import json
from anytree import Node, RenderTree
import pickle

# Download the JSON data
url = "https://oncotree.info/api/tumorTypes/tree"
response = requests.get(url)
data = response.json()

def build_tree(name, data):
    node = Node(name, fullname=data['name'], level=data['level'], maintype=data['mainType'], code=data['code'], tissue=data['tissue'])
    if "children" in data:
        for child_name, child_data in data["children"].items():
            child_node = build_tree(child_name, child_data)
            child_node.parent = node
    return node

# Build the tree starting from the root
root_name = "TISSUE"
root = build_tree(root_name, data[root_name])

# Print the tree structure
for pre, fill, node in RenderTree(root):
    print("%s%s\t%s" % (pre, node.name, node.fullname))

# Save the tree structure to a pickle file
with open('oncotree/oncotree.pkl', 'wb') as f:
    pickle.dump(root, f)
