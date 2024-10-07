## Files Format Description

The TSV files provided in this repository contain data used to evaluate the accuracy of our software pipeline for predicting protein–protein interactions, the data in the first four columns was used as an input for our fine-tuned Large Language Model (LLM), after the filtering of the pairs, predicted by our graph-based classification model as non-interacting. Each row in the TSV file represents a pair of proteins along with contextual information, the last row contains predictions made by our fine-tuned LLM.

### Column Breakdown

1. **Node IDs and Co-occurrence Value**:
   - **Format**: `<NodeID1> <NodeID2> <Co-occurrence Value>`
   - **Description**:
     - `NodeID1`: The unique identifier for the first protein, as defined in the ANDSystem/ANDDigest databases.
     - `NodeID2`: The unique identifier for the second protein, also from the ANDSystem/ANDDigest databases.
     - `Co-occurrence Value`: A numerical value representing the co-occurrence of the two proteins, calculated as \(1 - p\text{-value}\) from the ANDDigest database.

2. **Label of the First Protein**:
   - **Description**: The standard name or label of the first protein. In some examples this field can be presented as IntAct ID if the context was derived from the PubMed entries directly specified in the IntAct database as an evidence of an interaction between the pair. 

3. **Label of the Second Protein**:
   - **Description**: Similar to the second column, this is the standard name or label of the second protein.

4. **Prompt for the Language Model**:
   - **Description**: The input provided to our fine-tuned LLM. This prompt includes:
     - **Context**: Relevant information such as abstracts from PubMed articles that mention the proteins.
     - **Question**: A standardized question asking whether the two proteins interact.
     - **Instructions**: Guidelines for the LLM on how to format its response, including the two examples of interacting and non-interacting pairs. Since the LLM is aimed the predicting of `functional associations`, the examples were made using drug–drug interactions, to avoid bias toward specific protein interactions.

5. **LLM Prediction**:
   - **Description**: The output from our fine-tuned LLM, which includes:
     - **Answer**: A direct response of `YES` or `NO` indicating whether the proteins interact.
     - **Confidence Level**: An assessment of the confidence in the prediction (e.g., low, medium, high).
     - **Explanation**: A brief justification supporting the prediction.

### Example Entry

```
2A2861717 2A2459001 0.9999999209729    Vpu    CD4    [INST]Context: [PMID: 1727486 Human immunodeficiency virus type 1 Vpu protein regulates the formation of intracellular gp160-CD4 complexes...] Question: Based on the provided context, do the terms [Vpu protein] and [CD4 protein] interact with each other? NOTE: Always begin your response with 'YES' or 'NO'...[/INST]    YES, high confidence; Explanation: The context indicates that the Vpu protein and CD4 protein interact with each other...
```

#### Explanation of the Example

- **Node IDs and Co-occurrence Value**:
  - `2A2861717`: ANDSystem ID for the first protein (Vpu).
  - `2A2459001`: ANDSystem ID for the second protein (CD4).
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

### Notes

- **ANDSystem IDs**: Unique identifiers used within the ANDSystem database to reference specific proteins.

- **Co-occurrence Value**: Calculated as \(1 - p\text{-value}\) from the ANDDigest database, this value quantifies the statistical significance of the proteins being mentioned together in scientific literature.

- **Prompt Structure**:
  - **Context**: Offers necessary background information for the LLM to make an informed prediction.
  - **Question and Instructions**: Standardized format to ensure consistent LLM responses, including examples of interacting and non-interacting protein pairs.

- **LLM Predictions**: Designed to mimic human expert analysis by providing a clear answer, confidence level, and rationale.

### Usage

The set with positive examples was split into the two parts and archived, since the requirements to the uploding files size via the GitHub web-interface. In our study both parts were used for the assessement. The provided files can be reprocessed by the st6.LLM_eval.py, according to the  Usage section of the main page of  usage

### Additional Information

- **Fine-tuned LLM**: The language model has been specifically trained by us to interpret scientific texts and predict protein interactions based on contextual information. Is available via the HuggingFace repo (Timofey/Gemma-2-9b-it-Fused_PPI)[https://huggingface.co/Timofey/Gemma-2-9b-it-Fused_PPI/tree/main]

- **Data Sources**:
  - **ANDSystem/ANDDigest Databases**: Provides the protein IDs and co-occurrence statistics.
  - **IntAct Database** (accessed 2024-07-11): Offers curated interaction data and is used for validating the existing interactions. [Used query](https://www.ebi.ac.uk/intact/search?query=species:9606&interactorTypesFilter=protein&interactionTypesFil-ter=physical%20association,direct%20interaction&interactionHostOrganismsFilter=Homo%20sapiens)
  - **Stelzl2005 Dataset**: Offers curated interaction data and is used for validating the non-interactiong pairs. [link](http://www.russelllab.org/negatives/)
  - **PubMed Articles**: Source of the contextual information used in the prompts.
