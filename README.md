I apologize for that oversight. You're absolutely right - the parameter descriptions are important for users to understand how to use the scripts. Let me revise the README to include those critical details while maintaining the improved structure:

# Hybrid Approach for Protein Interaction Extraction from Scientific Literature

This repository contains the code and models for a hybrid approach to knowledge extraction from scientific publications, combining structured ontology models, pairwise co-occurrence, graph neural networks (GNNs), and large language models (LLMs).

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

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

1. Train the Graph Neural Network (GraphSAGE approach):

```bash
python st1.train_gnn.py --edges <path_to_input_edges_csv> --nodes <path_to_input_nodes_csv> --output <path_to_output_embeddings_csv>
```

Parameters:
- `--edges`: Path to the edges CSV file (default: `./graph_model/edges.csv`)
- `--nodes`: Path to the nodes CSV file (default: `./graph_model/nodes.csv`)
- `--output`: Path to save the generated node embeddings (default: `./graph_model/node_embeddings.128_64.csv`)

2. Train the Binary Classifier:

```bash
python st2.train_MLP_classifier.py --train <path_to_input_training_csv> --test <path_to_input_test_csv> --validation <path_to_input_validation_csv> --output <path_to_output_model_weights>
```

3. Fine-tune the Large Language Model:

   3.1. Fine-tune the model using mlx_lm:
   
   ```bash
   mlx_lm.lora --model <base_model_path> --train --data <training_dataset_path> --lora-layers -1 --iters 50000 --val-batches 1 --learning-rate 2.5e-5 --steps-per-report 250 --steps-per-eval 1000 --test --test-batches 1 --adapter-path <path_where_the_trained_LoRA_adapter_will_be_saved> --save-every 5000  --batch-size 1
   ```

   Parameters:
   - `--model`: Path to the base pre-trained LLM for fine-tuning (in our study, [google/Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) was used)
   - `--data`: Path to the dataset used in the fine-tuning process (available at [Timofey/protein_interactions_LLM_FT_dataset](https://huggingface.co/datasets/Timofey/protein_interactions_LLM_FT_dataset))

   3.2. Fuse the adapter with the base model:
   
   ```bash
   mlx_lm.fuse --model <base_model_path> --adapter-file <path_to_adapter> --save-path <fused_model_path> --de-quantize
   ```

   Parameters:
   - `--model`: Path to the base pre-trained LLM (same as in step 3.1)
   - `--adapter-file`: Path to the LoRA adapter obtained from step 3.1
   - `--save-path`: Path to save the fused model (The LLM used in our study is available at [Timofey/Gemma-2-9b-it-Fused_PPI](https://huggingface.co/Timofey/Gemma-2-9b-it-Fused_PPI))

## Data

- ANDSystem graph: Available upon request (examples in `./graph_model/` folder)
- MLP classifier weights: Available in `./MLP_classifier/` folder
- ANDDigest database: https://anddigest.sysbio.ru/
- Base LLM: [google/Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
- Fine-tuned and fused LLM: [Timofey/Gemma-2-9b-it-Fused_PPI](https://huggingface.co/Timofey/Gemma-2-9b-it-Fused_PPI)
- Fine-tuning dataset: [Timofey/protein_interactions_LLM_FT_dataset](https://huggingface.co/datasets/Timofey/protein_interactions_LLM_FT_dataset)

## Results

The pipeline achieved an accuracy of 0.772 (Matthews correlation coefficient) when evaluated on a corpus of experimentally confirmed protein interactions.

## Citation

If you use this code or models in your research, please cite our paper:

```
coming soon
```

## License

This project is licensed under the [MIT License].

## Acknowledgments

[List any acknowledgments or funding sources]

## Contact

For questions or support, please contact [itv@bionet.nsc.ru].
