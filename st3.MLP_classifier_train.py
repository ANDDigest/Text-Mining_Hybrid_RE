import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.inspection import permutation_importance

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a PPI MLP classifier.')
parser.add_argument(
    '--datasets_folder',
    type=str,
    default='./MLP_classifier/dataset/',
    help=('Path to the datasets folder containing '
          'st2.ppi_training_set.csv, st2.ppi_validation_set.csv, '
          'and st2.ppi_testing_set.csv.')
)
parser.add_argument(
    '--model_states_folder',
    type=str,
    default='./MLP_classifier/',
    help='Folder to save the trained model states.'
)
args = parser.parse_args()

# Load data from CSV files
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, 1:-1].values  # Ignore the first column (IDs) and the last column (labels)
    y = data.iloc[:, -1].values    # Last column as labels
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# Load train, validation, and test data
X_train, y_train = load_data(os.path.join(args.datasets_folder, 'st2.ppi_training_set.csv'))
X_val, y_val = load_data(os.path.join(args.datasets_folder, 'st2.ppi_validation_set.csv'))
X_test, y_test = load_data(os.path.join(args.datasets_folder, 'st2.ppi_testing_set.csv'))

# Combine train and validation data for final training
X_combined = torch.cat((X_train, X_val), dim=0)
y_combined = torch.cat((y_train, y_val), dim=0)

# Define the neural network model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(X_combined.shape[1], 256)
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

# Initialize the model, loss function, and optimizer
model = MLP()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Create DataLoader for batch processing
batch_size = 64
train_dataset = TensorDataset(X_combined, y_combined.float())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Early stopping criteria
early_stopping_patience = 50
best_mcc = -1
best_epoch = 0
best_model_state = None

# Train the model
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    epoch_train_outputs = []
    epoch_train_labels = []

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        epoch_train_outputs.extend(outputs.detach().cpu().numpy())
        epoch_train_labels.extend(y_batch.cpu().numpy())

    # Calculate training metrics
    epoch_train_loss /= len(train_loader)
    train_outputs = np.array(epoch_train_outputs)
    train_labels = np.array(epoch_train_labels)
    train_preds = (train_outputs > 0.5).astype(int)
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_precision = precision_score(train_labels, train_preds)
    train_recall = recall_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds)
    train_mcc = matthews_corrcoef(train_labels, train_preds)

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val).squeeze()
        y_val_pred = (val_outputs > 0.5).int()
        current_mcc = matthews_corrcoef(y_val, y_val_pred)
        current_accuracy = accuracy_score(y_val, y_val_pred)
        current_precision = precision_score(y_val, y_val_pred)
        current_recall = recall_score(y_val, y_val_pred)
        current_f1 = f1_score(y_val, y_val_pred)
        current_loss = criterion(val_outputs, y_val.float()).item()

    # Print metrics
    print(f'Epoch {epoch}:')
    print(f'Train Loss: {epoch_train_loss:.4f} Accuracy: {train_accuracy:.4f} Precision: {train_precision:.4f} Recall: {train_recall:.4f} F1 Score: {train_f1:.4f} MCC: {train_mcc:.4f}')
    print(f'Validation Loss: {current_loss:.4f} Accuracy: {current_accuracy:.4f} Precision: {current_precision:.4f} Recall: {current_recall:.4f} F1 Score: {current_f1:.4f} MCC: {current_mcc:.4f}')

    # Check for improvement in MCC
    if current_mcc > best_mcc:
        best_mcc = current_mcc
        best_epoch = epoch
        best_model_state = model.state_dict().copy()
        best_metrics = {
            "loss": current_loss,
            "accuracy": current_accuracy,
            "precision": current_precision,
            "recall": current_recall,
            "f1": current_f1,
            "mcc": current_mcc
        }
        print(f'MCC improved to {best_mcc:.4f}')
    else:
        print(f'MCC did not improve, current MCC: {current_mcc:.4f}')
    
    # Check early stopping condition
    if epoch - best_epoch >= early_stopping_patience:
        print(f'No improvement in MCC for {early_stopping_patience} epochs. Early stopping.')
        break

# Load the best model state
model.load_state_dict(best_model_state)

# Evaluate the best model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test).squeeze()
    y_pred = (test_outputs > 0.5).int()

# Calculate metrics for the test set
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_mcc = matthews_corrcoef(y_test, y_pred)

# Print the metrics for the best model and the epoch it was obtained
print(f'Best Model obtained at Epoch: {best_epoch}')
print('Best Model Metrics on Validation Set:')
print(f'Validation Loss: {best_metrics["loss"]:.4f}')
print(f'Accuracy: {best_metrics["accuracy"]:.4f}')
print(f'Precision: {best_metrics["precision"]:.4f}')
print(f'Recall: {best_metrics["recall"]:.4f}')
print(f'F1 Score: {best_metrics["f1"]:.4f}')
print(f'MCC: {best_metrics["mcc"]:.4f}')

# Print the calculated metrics for the test set
print('Best Model Metrics on Test Set:')
print(f'Accuracy: {test_accuracy:.4f}')
print(f'Precision: {test_precision:.4f}')
print(f'Recall: {test_recall:.4f}')
print(f'F1 Score: {test_f1:.4f}')
print(f'MCC: {test_mcc:.4f}')

# Ensure the model_states_folder exists
os.makedirs(args.model_states_folder, exist_ok=True)

# Save the best model to a .pth file
torch.save(model.state_dict(), os.path.join(args.model_states_folder, 'PPI_mlp_model.pth'))

# Custom wrapper class for the model
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        pass  # No fitting necessary for the wrapped model

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
        return (outputs > 0.5).astype(int)

# Compute feature importance using permutation importance
wrapped_model = ModelWrapper(model)
results = permutation_importance(
    wrapped_model, X_test.numpy(), y_test.numpy(), scoring='accuracy', n_repeats=10, random_state=42
)
importance_means, importance_stds = results.importances_mean, results.importances_std

# Print feature importances
print('Feature Importances:')
for i, (mean, std) in enumerate(zip(importance_means, importance_stds)):
    print(f'Feature {i}: Mean Importance = {mean:.4f}, Std = {std:.4f}')
