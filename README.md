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
cd BreadcrumbsText-Mining_Hybrid_RE
pip install -r requirements.txt
```

## Usage

1. Train the Graph Neural Network:
```python
python train_gnn.py
```

2. Train the Binary Classifier:
```python
python train_classifier.py
```

3. Fine-tune the Large Language Model:
```python
python finetune_llm.py
```

4. Run the full pipeline:
```python
python run_pipeline.py
```

## Data

- The ANDSystem graph is available upon request
- ANDDigest database: https://anddigest.sysbio.ru/
- Fine-tuned LLM: https://huggingface.co/Timofey/Gemma-2-9b-it-Fused_PPI
- Training dataset: https://huggingface.co/Timofey/Gemma-2-9b-it-Fused_PPI

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
