import pandas as pd
import torch
import torch.nn as nn
import argparse

# Parse command-line arguments with default values
parser = argparse.ArgumentParser(description='Run MLP model on input data.')
parser.add_argument('--input_file_path', type=str, default='./validation/intact_positive_PPI_2024-07-11-08-09.GNN_input.csv',
                    help='Path to the input CSV file.')
parser.add_argument('--model_path', type=str, default='./MLP_classifier/PPI_mlp_model.pth',
                    help='Path to the saved PyTorch model.')
parser.add_argument('--output_file_path', type=str, default='./validation/intact_positive_PPI_2024-07-11-08-09.GNN_output.csv',
                    help='Path to save the output CSV file.')

args = parser.parse_args()

input_file_path = args.input_file_path
model_path = args.model_path
output_file_path = args.output_file_path

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

# Load and preprocess the input data
data = pd.read_csv(input_file_path, header=None)
ids = data.iloc[:, 0].values  # First column as IDs
X = data.iloc[:, 1:].values   # All columns except the first as input features

# Load the saved model
model = MLP()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Convert input data to torch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)

# Perform predictions
with torch.no_grad():
    outputs = model(X_tensor).squeeze()
    predictions = (outputs > 0.5).int().numpy()

# Save the results to a CSV file
output_df = pd.DataFrame({'ID': ids, 'Prediction': predictions})
output_df.to_csv(output_file_path, index=False)

print(f'Predictions saved to {output_file_path}')
