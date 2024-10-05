This folder contains the trained binary classification model based on the Multi Layer Perceptron with three hidden layers.

## Content

### `PPI_mlp_model.pth`
`PPI_mlp_model.pth` - the weights in the PyTorch serialized tensors format of our MultiLayer Perceptron (MLP) model, trained for predicting edges between the protein pairs based on the Graph structure. These weights were utilized in our study.

### `dataset` Directory

The `dataset` folder contains sample files demonstrating the format of the dataset used for training the MultiLayer Perceptron (MLP) model. The complete dataset is divided into three subsets: training, testing, and validation.

#### File Structure
Each subset is represented by a CSV file with the following column structure:

1. **First column**: Contains the ANDSystem IDs of the source and target proteins, separated by an underscore (`_`). These IDs correspond to the last column in `../graph_model/nodes.csv`.
2. **Columns 2-65**: 64 comma-separated values representing the vector features of the first node's embeddings in `../graph_model/node_embeddings.128_64.csv`.
3. **Columns 66-129**: 64 comma-separated values representing the vector features of the second node's embeddings in `../graph_model/node_embeddings.128_64.csv`.
4. **Column 130**: Co-occurrence value, calculated as `(1 - p-value)` for positive examaples, and a random float in the range `(0,1)` for the `30%` of negatives.
5. **Last column**: Label indicating the presence (`1`) or absence (`0`) of an edge between the proteins in `../graph_model/edges.csv`.

> [!NOTE]
> #### Note on Dataset Samples
> Due to file size constraints, only small samples of the dataset are provided in this repository. These samples are intended to illustrate the data format and structure. The complete dataset used in our study is available upon request.
