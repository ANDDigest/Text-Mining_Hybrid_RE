This folder contains the examples, only for demonstration purposes, of the format, used to train a Graph Neural Network (GNN) using the GraphSAGE approach to obtain vector representations (embeddings) of graph nodes. The graph is represented by two separate CSV tables: `nodes.csv` and `edges.csv`.

## Input Data

### 1. `nodes.csv`
The `nodes.csv` file contains information about the nodes in the graph. Each row corresponds to a single node and follows this structure:

- **First column**: An integer autoincrement ID used by the training script, starting from 0.
- **Next columns**: These columns contain mostly zeros, with exactly one column containing a `1` value. The position of the `1` indicates the type of the node.
- **Last column**: This is the object's unique ID in the ANDDigest/ANDSystem databases.

### 2. `edges.csv`
The `edges.csv` file contains the information about the edges (connections) between nodes. Each row follows this structure:

- **First column**: The edge's unique ID.
- **Second column**: The ID of the source node from `nodes.csv`.
- **Third column**: The ID of the target node from `nodes.csv`.
- **Next columns**: These columns indicate the interaction types between the two nodes. Unlike `nodes.csv`, an edge can have multiple `1` values in these columns, reflecting multiple interaction types. (Note: This information is not used in the GraphSAGE approach.)
- **Last column**: The (1 - co-occurrence p-value) of the node pair according to their occurence in PubMed texts.

## Output Data

The output of the training process is a CSV file containing the 64-dimensional vector (embedding) for each node in the graph. This vector represents the node's learned features, used for further training of the MLP classifier for the graph-based protein interaction prediction task.
