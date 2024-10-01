# Hybrid Approach, combining Structured Ontology Models, Pairwise Co-occurrence, Graph Neural Networks, and Large Language Models, to Extraction of Protein Interactions from Scientific Literature

This repository contains the code and models for a hybrid approach to knowledge extraction from scientific publications using structured ontology models, graph neural networks (GNNs), and large language models (LLMs).

## Overview

Our method combines text-mining techniques with GNNs and fine-tuned LLMs to extend biomedical knowledge graphs and interpret predicted edges based on published literature. The approach achieves high accuracy in predicting protein-protein interactions and can be applied to analyze complex disorders like insomnia.

Key components:
- Graph Neural Network for learning node embeddings
- Binary classifier based on a Multilayer Perceptron 
- Fine-tuned Large Language Model for validating predictions in the literature

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.13.0+
- PyTorch Geometric 2.5.3+
- MLX-LM 0.16.1+
- Hugging Face Transformers
- Scikit-learn 1.5.0+
- Pandas

### Installation

```bash
git clone https://github.com/ANDDigest/Text-Mining_Hybrid_RE.git
cd ./Text-Mining_Hybrid_RE
pip install -r requirements.txt
```

## Usage

1. Train the Graph Neural Network (GraphSAGE approach) to obtain vector representations of graph nodes:
```bash
python st1.train_gnn.py --edges <path_to_input_edges_csv> --nodes <path_to_input_nodes_csv> --output <path_to_output_embeddings_csv>
```

- `--edges`: Path to the edges CSV file (default: `./graph_model/edges.csv`)
- `--nodes`: Path to the nodes CSV file (default: `./graph_model/nodes.csv`)
- `--output`: Path to save the generated node embeddings (default: `./graph_model/node_embeddings.128_64.csv`)

2. Train the Binary Classifier:
```bash
python st2.train_MLP_classifier.py --train <path_to_input_training_csv> --test <path_to_input_test_csv> --validation <path_to_input_validation_csv> --output <path_to_output_model_weights>
```

3. Fine-tune the Large Language Model:<br>

    3.1. The fine-tuning process was performed under MAC OS, with a following parameters, using the [mlx_lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm):
    - `--model`: Path to the base pre-trained LLM for fine-tuning (in our study the [google/Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) was used);
    - `--data`: Path to the dataset used in the fine-tuning process (available at [Timofey/protein_interactions_LLM_FT_dataset](https://huggingface.co/datasets/Timofey/protein_interactions_LLM_FT_dataset).
<br>
    ```bash
    mlx_lm.lora --model <base_model_path> --train --data <training_dataset_path> --lora-layers -1 --iters 50000 --val-batches 1 --learning-rate 2.5e-5 --steps-per-report 250 --steps-per-eval 1000 --test --test-        batches 1 --adapter-path <path_where_the_trained_LoRA_adapter_will_be_saved> --save-every 5000  --batch-size 1
    ```

    3.2. The obtained adapter was fused with the base model, using the following shell command:
    ```bash
    mlx_lm.fuse --model <base_model_path> --adapter-file <path_to_adapter> --save-path <fused_model_path> --de-quantize
    ```
    - `--model`: Path to the base pre-trained LLM for fine-tuning (in our study the [google/Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) was used);
    - `--adapter-file`: Path to the LoRA adapter, obtained by performing command from 3.1.;
    - `--save-path`: Path to where the fused model is saved. The LLM, used in our study, for the context-based protein interaction prediction is available at [Timofey/Gemma-2-9b-it-Fused_PPI](https://huggingface.co/Timofey/Gemma-2-9b-it-Fused_PPI).

## Data

- The ANDSystem graph is available upon request, the examples of input and output formats with description, can be found in the `./graph_model/` folder. The entire graph and MLP classifier training sets used in the study, due to the large size, are available upon request;
- MLP weights for the protein classification task, used in the study, are available in the `./MLP_classifier` folder of this repository;
- ANDDigest database: https://anddigest.sysbio.ru/;
- Base LLM, used for the fine-tuning: [google/Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
- Fine-tuned and fused LLM: [Timofey/Gemma-2-9b-it-Fused_PPI](https://huggingface.co/Timofey/Gemma-2-9b-it-Fused_PPI)
- Dataset, used for the LLM fine-tuning: [Timofey/protein_interactions_LLM_FT_dataset](https://huggingface.co/datasets/Timofey/protein_interactions_LLM_FT_dataset)

## Results

The pipeline achieved an accuracy of 0.772 (measured as the Matthews correlation coefficient) when evaluated on a corpus of experimentally confirmed protein interactions.

## Citation

If you use this code or models in your research, please cite our paper:

```
coming soon
```

## License

This project is licensed under the [MIT License].

## Acknowledgments

- [List any acknowledgments or funding sources]

## Contact

For questions or support, please contact [itv@bionet.nsc.ru].
