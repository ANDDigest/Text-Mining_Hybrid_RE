# Files Format Description

The TSV files provided in this repository contain archived data used to evaluate the accuracy of our software pipeline for predicting protein–protein interactions for the pairs, predicted by our binary classification model as interacting according to the human interactome graph topology (subsections `2.5.`, `3.1.4.` of our manuscript). The data inside the `GoldStandards` subfolder was used for computing the accuracy of our fine-tuned model based on known gold standards (subsections `2.4.`, `3.1.5` of our manuscript). Due to size limitations, the positive set was split into two files and archived. In our study, both archived TSV files with positive examples were used.

Each row in the TSV file represents a pair of proteins along with contextual information and predictions made by our fine-tuned Language Model (LLM).

## Column Breakdown

1. **Node IDs and Co-occurrence Value**:
   - **Format**: `<NodeID1> <NodeID2> <Co-occurrence Value>`
   - **Description**:
     - `NodeID1`: The unique identifier for the first protein, as defined in the ANDSystem/ANDDigest databases.
     - `NodeID2`: The unique identifier for the second protein, also from the ANDSystem/ANDDigest databases.
     - `Co-occurrence Value`: A numerical value representing the co-occurrence of the two proteins, calculated as $1 - p\text{-value}$ from the ANDDigest database.

2. **Label of the First Protein**:
   - **Description**: The standard name or label of the first protein. In cases where PMID was extracted directly from the IntAct database as an experimental study where the interaction was first described, this may be the UniProtKB, DIP, or IntAct ID of the object.

3. **Label of the Second Protein**:
   - **Description**: Similar to the second column, this is the standard name or label of the second protein.

4. **Prompt for the Language Model**:
   - **Description**: The input provided to our fine-tuned LLM. This prompt includes:
     - **Context**: Relevant information such as abstracts from PubMed articles that mention the proteins.
     - **Question**: A standardized question asking whether the two proteins interact.
     - **Instructions**: Guidelines for the LLM on how to format its response, including examples of interacting and non-interacting pairs, using drug–drug interactions. This was done to avoid bias toward specific protein interactions. 

5. **LLM Prediction**:
   - **Description**: The output from our fine-tuned LLM, which includes:
     - **Answer**: A direct response of `YES` or `NO` indicating whether the proteins interact.
     - **Confidence Level**: An assessment of the confidence in the prediction (e.g., low, medium, high).
     - **Explanation**: A brief justification supporting the prediction.

## Example Entry

```
2A2861717 2A2459001 0.9999999209729    Vpu    CD4    [INST]Context: [PMID: 1727486 Human immunodeficiency virus type 1 Vpu protein regulates the formation of intracellular gp160-CD4 complexes...] Question: Based on the provided context, do the terms [Vpu protein] and [CD4 protein] interact with each other? NOTE: Always begin your response with 'YES' or 'NO'...[/INST]    YES, high confidence; Explanation: The context indicates that the Vpu protein and CD4 protein interact with each other...
```

### Explanation of the Example

- **Node IDs and Co-occurrence Value**:
  - `2A2861717`: ANDSystem/ANDDigest ID for the first protein (Vpu).
  - `2A2459001`: ANDSystem/ANDDigest ID for the second protein (CD4).
  - `0.9999999209729`: High co-occurrence value indicating a significant association between the two proteins.

- **Label of the First Protein**:
  - `Vpu`: The name of the first protein.

- **Label of the Second Protein**:
  - `CD4`: The name of the second protein.

- **Prompt for the Language Model**:
  - Enclosed within `[INST]` and `[/INST]`, the prompt provides context from a PubMed article (e.g., PMID: 1727486) discussing the interaction between Vpu and CD4.
  - The question explicitly asks if the two proteins interact, along with instructions and examples to guide the LLM's response.

- **LLM Prediction**:
  - `YES, high confidence; Explanation: The context indicates that the Vpu protein and CD4 protein interact with each other...`
  - The LLM confirms the interaction with high confidence and provides an explanation based on the provided context.

## Notes

- **ANDSystem IDs**: Unique identifiers used within the ANDSystem database to reference specific proteins.

- **Co-occurrence Value**: Calculated as $1 - p\text{-value}$ from the ANDDigest database, this value quantifies the statistical significance of the proteins being mentioned together in scientific literature.

- **Prompt Structure**:
  - **Context**: Offers necessary background information for the LLM to make an informed prediction.
  - **Question and Instructions**: Standardized format to ensure consistent LLM responses, including examples of interacting and non-interacting protein pairs.

- **LLM Predictions**: Designed to mimic human expert analysis by providing a clear answer, confidence level, and rationale.

## Usage

The results obtained in our study with the fine-tuned LLM module can be replicated using the provided data files.

- **Data Analysis**: Each row can be parsed to extract the proteins of interest, their interaction context, and the LLM's prediction for further analysis or validation.

- **Model Evaluation**: Compare the LLM's predictions against known interactions to assess model performance.

- **Research and Development**: Use the prompts and LLM outputs as a basis for developing improved models or for understanding the nuances in protein–protein interaction predictions.

## Additional Information

- **Fine-tuned LLM**: The language model has been specifically trained to interpret scientific texts and predict protein interactions based on contextual information in the input format provided in the TSV files.

- The files starting with `IntAct` were formed based on pairs of proteins experimentally shown as interacting, obtained from a query to the IntAct web interface, after pre-filtering with our graph-based binary classifier (i.e., `../MLP_classifier/PPI_mlp_model.pth`).
- The file begining with `16169070` was formed based on the subset of non-interacting protein pairs from the Stelzl2005 dataset, also after its pre-filtering with our binary classifier.  

- **Data Sources**:
  - **ANDSystem/ANDDigest Databases**: Provides the human interactome graph, protein IDs, labels, and co-occurrence statistics.
  - **IntAct Database** (accessed 07-11-2023): Offers curated interaction data and is used for validating the interactions on positive examples. [Used query](https://www.ebi.ac.uk/intact/search?query=species:9606&interactorTypesFilter=protein&interactionTypesFilter=physical%20association,direct%20interaction&interactionHostOrganismsFilter=Homo%20sapiens)
  - **Stelzl2005 Dataset** (accessed 04-10-2023): Offers curated data for human protein pairs shown in experiments as non-interacting, and is used for validating the interactions on negative (non-interacting) examples. [Link](http://www.russelllab.org/negatives/)
  - **PubMed Articles**: Source of the contextual information used in the prompts.

- **Ethical Considerations**: All data used complies with relevant data usage policies and respects intellectual property rights.
