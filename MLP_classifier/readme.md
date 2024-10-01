This folder contains the archive with a dataset of protein embeddings, obtained by the performing of `st1.train_gnn.py` script from the main folder. for the training of the classification model based on the Multi Layer Perceptron with three hidden layers.

## Content

### 1. `dataset_MLP.zip`
The archive contains dataset, split into the training, validation and testing files. Each of the files is presented by the comma-separated table in the .csv format with a following structure:

- **First column**: a pair of unique protein IDs from the ANDDigest/ANDSystem databases, corresponding to the `nodes.csv` of the graph, separated with the `_` sign;
- **Next columns**: The next 64 columns contain numeric vector representation of the first node from the ANDSystem Graph, the next 64 columns correspond to the second node.
- **Last two columns**: Two last columns are the level of confidence (1 - pairwise co-occurence <i>p</i>-value) and the label indicating if it is a positive (1), or negative (0) example.

### 2. `PPI_mlp_model.pth`
The `PPI_mlp_model.pth` are the weights in the PyTorch serialized tensors format, of our MultiLayer Perceptron (MLP) model, trained for predicting edges between the protein pairs based on the Graph structure. These weights were utilized in our study.
