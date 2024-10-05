This folder contains the trained binary classification model based on the Multi Layer Perceptron with three hidden layers.

## Content

### `PPI_mlp_model.pth`
`PPI_mlp_model.pth` - the weights in the PyTorch serialized tensors format of our MultiLayer Perceptron (MLP) model, trained for predicting edges between the protein pairs based on the Graph structure. These weights were utilized in our study.

### `dataset`
`dataset` - the folder contains examples of the dataset format used in the training of MultiLayer Perceptron (MLP) model. The dataset is split into three sets: training, testing, and validation. Each set is presented by the CSV table with a comma-separated columns, where the first column contains ANDSystem's IDs of the source and the target proteins, corresponding to the '../graph_model/node_embeddings.128_64.csv' divided by the `_` symbol. ...

Content:
- `st2.ppi_testing_set.csv`: testing set examples
- `--input_edges`: Path to the edges CSV file (default: `./graph_model/nodes.csv`)
- `--input_embeddings`: Path to the node embeddings obtained in **step 1** (default: `./graph_model/node_embeddings.128_64.csv`)
- `--output_ppi_learning_set`: Path where the generated training, validation and testing sets are saved (default: `./graph_model/mlp_dataset/`)
