import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse

def escape_prompt(prompt):
    # Remove newlines and escape double quotes
    prompt = prompt.replace('\n', '').replace('"', '\\"').replace(' . ', '. ')
    return prompt

def main(input_tsv_file, output_tsv_file, model_name):
    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
        device_index = 0  # You can change this if you have multiple GPUs
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Metal Performance Shaders) on Apple Silicon")
        device_index = 0  # MPS uses device index 0
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
        device_index = -1  # CPU device index

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Set up the pipeline for text generation
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_index
    )

    with open(input_tsv_file, 'r', newline='') as infile, open(output_tsv_file, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')

        for row in reader:
            if len(row) >= 4:
                prompt = '<s>' + escape_prompt(row[3])
                # Generate text
                response = generator(prompt, max_new_tokens=256)[0]['generated_text']

                # Extract only the newly generated text
                new_text = response[len(prompt):].strip()

                print(new_text)
                row.append(new_text)
                writer.writerow(row)
            else:
                print("Row has fewer than 4 columns, skipping:", row)

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Process a TSV file using a HuggingFace Transformer model.')
    parser.add_argument('--input_tsv_file', type=str, required=True,
                        help='Path to the input TSV file.')
    parser.add_argument('--output_tsv_file', type=str, required=True,
                        help='Path to the output TSV file.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name or path of the HuggingFace Transformer model.')

    args = parser.parse_args()

    main(args.input_tsv_file, args.output_tsv_file, args.model_name)
