import pandas as pd
import torch
import torch.nn as nn
import argparse

# Define the neural network model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(129, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.sigmoid(self.output(x))
        return x

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP classifier on input data.")
    parser.add_argument('--input_file', type=str, default='./MLP_classifier/validation/MLP_negative.stelzl2005.input.csv', 
                        help='Path to the input CSV file.')
    parser.add_argument('--model_file', type=str, default='./MLP_classifier/PPI_mlp_model.pth', 
                        help='Path to the saved model file.')
    parser.add_argument('--output_file', type=str, default='./MLP_classifier/validation/MLP_negative.stelzl2005.output.csv', 
                        help='Path to the output CSV file.')
    return parser.parse_args()

# Main function
def main():
    # Get command-line arguments
    args = parse_args()

    # Load and preprocess the input data
    input_file_path = args.input_file
    data = pd.read_csv(input_file_path, header=None)
    ids = data.iloc[:, 0].values  # First column as IDs
    X = data.iloc[:, 1:].values  # All columns except the first as input features

    # Load the saved model
    model = MLP()
    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    # Convert input data to torch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Perform predictions
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        predictions = (outputs > 0.5).int().numpy()

    # Save the results to a CSV file
    output_df = pd.DataFrame({'ID': ids, 'Prediction': predictions})
    output_file_path = args.output_file
    output_df.to_csv(output_file_path, index=False)

    print(f'Predictions saved to {output_file_path}')

if __name__ == "__main__":
    main()
