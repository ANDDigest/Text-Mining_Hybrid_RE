# Hybrid approach to knowledge extraction from scientific publications using structured ontology models, graph neural networks, and large language models

This repository contains the code and models for a hybrid approach to knowledge extraction from scientific publications, combining structured ontology models, pairwise co-occurrence, graph neural networks (GNNs), and large language models (LLMs).

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Training](#training)
- [Usage](#usage)
- [Data](#data)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Overview

Our method combines text-mining techniques with GNNs and fine-tuned LLMs to extend biomedical knowledge graphs and interpret predicted edges based on published literature. The approach achieves high accuracy in predicting protein-protein interactions and can be applied to analyze complex disorders like insomnia.

### Key components:
- Graph Neural Network for learning node embeddings
- Binary classifier based on a Multilayer Perceptron 
- Fine-tuned Large Language Model for validating predictions in the literature

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.13.0+
- PyTorch Geometric 2.5.3+
- MLX-LM 0.16.1+
- Transformers 4.42.4+
- Scikit-learn 1.5.0+
- Pandas
- NumPy

### Installation

```bash
git clone https://github.com/ANDDigest/Text-Mining_Hybrid_RE.git
cd ./Text-Mining_Hybrid_RE
pip install -r requirements.txt
```

## Training

**1. Train the Graph Neural Network (GraphSAGE approach):**

```bash
python st1.train_gnn.py --edges /path/to/edges_csv --nodes /path/to/nodes.csv --output /path/to/output_embeddings.csv
```

Parameters:
- `--edges`: Path to the edges CSV file (default: `./graph_model/edges.csv`)
- `--nodes`: Path to the nodes CSV file (default: `./graph_model/nodes.csv`)
- `--output`: Path to save the generated node embeddings (default: `./graph_model/node_embeddings.128_64.csv`)

**2. Embeddings-based MLP learning set formation:**

```bash
perl st2.MLP_dataset_gen.pl --input_nodes=./path/to/nodes.csv --input_edges=./path/to/edges.csv --input_embeddings=./path/to/embeddings.csv --output_ppi_learning_set=./path/to/output/
```

Parameters:
- `--input_nodes`: Path to the nodes CSV file (default: `./graph_model/edges.csv`)
- `--input_edges`: Path to the edges CSV file (default: `./graph_model/nodes.csv`)
- `--input_embeddings`: Path to the node embeddings obtained in **step 1** (default: `./graph_model/node_embeddings.128_64.csv`)
- `--output_ppi_learning_set`: Path where the generated training, validation and testing sets are saved (default: `./graph_model/mlp_dataset/`)

> [!NOTE]
> Our study utilized a comprehensive training dataset consisting of 460,000 pairwise protein interactions, derived from the vector representations of corresponding nodes in the ANDSystem graph. This dataset was carefully divided into three subsets: a training set with 200,000 positive and 200,000 negative examples, a validation set containing 20,000 examples of each type, and a testing set with 10,000 examples of each type. Positive examples were defined as pairs of proteins connected by an edge in the [ANDSystem](https://link.springer.com/article/10.1186/s12859-018-2567-6) graph, while negative examples were pairs without such connections. Due to the substantial size of these dataset files, we have included only a small sample in the `./MLP_classifier/dataset/` directory for reference. However, the complete original version used in our manuscript can be available upon request.

**3. Train the Binary MLP Classifier:**

```bash
python st3.MLP_classifier_train.py --datasets_folder /path/to/datasets_folder --model_states_folder /path/to/save_model
```

Parameters:
- `--datasets_folder`: Directory containing the dataset files (default: `./MLP_classifier/dataset/`)
  The script expects the following three files in this directory:
  - `st2.ppi_training_set.csv`
  - `st2.ppi_validation_set.csv`
  - `st2.ppi_testing_set.csv`

- `--model_states_folder`: Directory where the trained model will be saved (default: `./MLP_classifier/`). The trained model will be saved as `PPI_mlp_model.pth`.

> [!IMPORTANT]
> The file `./MLP_classifier/PPI_mlp_model.pth` in this repository contains the weights of the trained classification model used in our research for predicting interactions between protein pairs.

**4. Fine-tune the Large Language Model:**

   <i>4.1. Fine-tune the model using mlx_lm:</i>
   
   ```bash
   mlx_lm.lora --model <base_model_path> --train --data <training_dataset_path> --lora-layers -1 --iters 50000 --val-batches 1 --learning-rate 2.5e-5 --steps-per-report 250 --steps-per-eval 1000 --test --test-batches 1 --adapter-path <path_where_the_trained_LoRA_adapter_will_be_saved> --save-every 5000  --batch-size 1
   ```

   Parameters:
   - `--model`: Path to the base pre-trained LLM for fine-tuning (in our study, [google/Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) was used)
   - `--data`: Path to the dataset for using in the fine-tuning process (available at huggingface [Timofey/protein_interactions_LLM_FT_dataset](https://huggingface.co/datasets/Timofey/protein_interactions_LLM_FT_dataset))

   <i>4.2. Fuse the adapter with the base model:</i>
   
   ```bash
   mlx_lm.fuse --model <base_model_path> --adapter-file <path_to_adapter> --save-path <fused_model_path> --de-quantize
   ```

   Parameters:
   - `--model`: Path to the base pre-trained LLM (same as in `step 3.1`)
   - `--adapter-file`: Path to the LoRA adapter obtained from `step 3.1`
   - `--save-path`: Path to save the fused model (The Fused LLM, used in our study, is available at huggingface [Timofey/Gemma-2-9b-it-Fused_PPI](https://huggingface.co/Timofey/Gemma-2-9b-it-Fused_PPI))

## Usage

**1. MLP Model for Edges Classification:**

```bash
python st5.MLP_eval.py  \
    --input_file_path '/path/to/vectorized_protein_pairs.csv' \
    --model_path '/path/to/MLP_model.pth' \
    --output_file_path '/path/to/output_file.csv'
```

Parameters:
- `--input_file_path`: Path to the file with vectorized node pairs in the CSV format (default: `./validation/intact_positive_PPI_2024-07-11-08-09.GNN_input.csv`)
- `--model_path`: Path to the trained MLP binary classification (default: `./MLP_classifier/PPI_mlp_model.pth`)
- `--output`: Path to the graph-based classification model prediction results (default: `./validation/intact_positive_PPI_2024-07-11-08-09.GNN_output.csv`)

**2. LLM for Context-based Edges Classification:**

```bash
python st6.LLM_eval.py  \
    --input_tsv_file '/path/to/input_file.tsv' \
    --output_tsv_file '/path/to/output_file.tsv' \
    --model_name 'model_name_or_path'
```

Parameters:
- `--input_tsv_file`: path to the input TSV file, containing the list of prompts (default: `./validation/context_based/16169070_PPI.LLM_input_and_output.tsv`)
- `--output_tsv_file`: path to the output TSV file, containing the column with model outputs (default: `./validation/context_based/16169070_PPI.LLM_output.tsv`)
- `--model_name`: Name or local path of the HuggingFace Transformer model to use (default: `Timofey/Gemma-2-9b-it-Fused_PPI`).

>[!IMPORTANT]
>The script expects each row in the input TSV file to have at least four columns. The prompt to process is assumed to be in the fourth column (`row[3]`). If a row has fewer than four columns, it is skipped with a warning message. The content inside the other columns, e.g., `rows[0-2]` or `row[4]`, isn't directly used by the model, and these fields can be left empty if needed. More detailed information about the format, as well as validation examples used in our study, available at `./validation/context_based/`

## Validation Datasets

> [!NOTE]
> Detailed description of input formats of data for our graph-based binary classifier (`to be added 08.10.2024`) and LLM model, are available inside the corresponding parts of the `./validation/` folder

## Data

- ANDSystem graph, node embeddings and MLP full training dataset: Available upon request (examples in `./graph_model/` and `./MLP_classifier/dataset/` folders)
- Graph-based MLP binary classification model, used in our study: Available in `./MLP_classifier/` folder
- ANDDigest database: [link](https://anddigest.sysbio.ru/)
- LLM, used in our study as a base model for fine-tuning: [google/Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
- Our fine-tuning dataset: [Timofey/protein_interactions_LLM_FT_dataset](https://huggingface.co/datasets/Timofey/protein_interactions_LLM_FT_dataset)
- Our fine-tuned and fused LLM: [Timofey/Gemma-2-9b-it-Fused_PPI](https://huggingface.co/Timofey/Gemma-2-9b-it-Fused_PPI)
- List of human protein pairs, experimentally shown as non-interacting: [Stelzl2005](http://www.russelllab.org/negatives/)

## Results

The method achieved an accuracy of `0.772` (Matthews correlation coefficient) when evaluated on a corpus of experimentally confirmed protein interactions.

## Citation

If you use this code or models in your research, please cite our paper:

```
Ivanisenko, T.V.; Demenkov, P.S.; Ivanisenko, V.A. An Accurate and Efficient Approach to Knowledge Extraction from Scientific Publica-tions Using Structured Ontology Models, Graph Neural Networks, and Large Language Models. Int. J. Mol. Sci. 2024, 24, 25, 11811. https://doi.org/10.3390/ijms252111811
```

## License

This project is licensed under the [MIT License].

## Acknowledgments

This work was supported by a grant for research centers, provided by the Analytical Center for the Government of the Russian Federation in accordance with the subsidy agreement (agreement identifier 000000D730324P540002) and the agreement with the Novosibirsk State University dated December 27, 2023 No. 70-2023-001318.

## Contact

For questions or support, please contact [itv@bionet.nsc.ru].
