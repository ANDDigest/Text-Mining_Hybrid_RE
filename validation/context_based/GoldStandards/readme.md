# GoldStandards

This folder contains datasets used to evaluate the accuracy of our fine-tuned Large Language Model (LLM) in predicting protein–protein interactions based on textual data. These datasets are derived from established gold standard corpora and include both positive and negative examples.

## Contents

- `HPRD50.LLM_input_and_output.pos.tsv`
- `HPRD50.LLM_input_and_output.neg.tsv`
- `IEPA.LLM_input_and_output.pos.tsv`
- `IEPA.LLM_input_and_output.neg.tsv`

## Description

### HPRD50 Datasets

The HPRD50 datasets are based on the HPRD50 gold standard corpus from [bigbio/hprd50](https://huggingface.co/datasets/bigbio/hprd50), which comprises manually annotated abstracts from biomedical publications describing various protein–protein interactions.

- **HPRD50.LLM_input_and_output.pos.tsv**: Contains positive examples where the abstracts explicitly describe interactions between protein pairs.
- **HPRD50.LLM_input_and_output.neg.tsv**: Contains negative examples featuring protein pairs known not to interact, along with abstracts mentioning these proteins.

**Reference:**
`Fundel, K., Küffner, R., & Zimmer, R. (2007). RelEx—Relation extraction using dependency parse trees. Bioinformatics, 23(3), 365–371. https://doi.org/10.1093/bioinformatics/btl616`

### IEPA Datasets

The IEPA datasets are derived from the Interaction Extraction Performance Assessment (IEPA) corpus [bigbio/iepa](https://huggingface.co/datasets/bigbio/iepa), known for its manually annotated documents detailing protein interactions.

- **IEPA.LLM_input_and_output.pos.tsv**: Includes positive examples of interacting protein pairs with associated abstracts.
- **IEPA.LLM_input_and_output.neg.tsv**: Contains negative examples of non-interacting protein pairs and their corresponding abstracts.

**Reference:**
`Ding, J., Berleant, D., Nettleton, D., & Wurtele, E. (2001). Mining MEDLINE: abstracts, sentences, or phrases? In R. B. Altman, A. K. Dunker, L. Hunter, K. Lauderdale, & T. E. Klein (Eds.), Biocomputing 2002 (pp. 326–337). WORLD SCIENTIFIC. https://doi.org/10.1142/9789812799623_0031`

## Usage

These datasets were used in our study (subsections `2.4.` and `3.1.4.` of the manuscript) to assess the performance of the fine-tuned LLM in predicting protein–protein interactions from textual information. Each file follows the same format as described in the main [Files Format Description](../readme.md), which details the structure of the input and output data for the LLM.

## Notes

- Each dataset contains an equal number of positive and negative examples to ensure unbiased evaluation.
- **Data Sources**: The positive examples are based on known interactions from the `HPRD50` and `IEPA` corpora, while the negative examples are derived from protein pairs known not to interact, from `Stelzl2005` list.
- **File Format**: The TSV files include document ids and protein identifiers, labels, contextual prompts extracted from PubMed abstracts, and the our LLM's predictions.
