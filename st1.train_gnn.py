import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import argparse

# Argument parser for loading and saving paths
parser = argparse.ArgumentParser(description='GraphSAGE Model Training')
parser.add_argument('--edges', type=str, default='./graph_model/edges.csv', help='Path to the edges CSV file')
parser.add_argument('--nodes', type=str, default='./graph_model/nodes.csv', help='Path to the nodes CSV file')
parser.add_argument('--output', type=str, default='./graph_model/node_embeddings.128_64.csv', help='Output path for node embeddings')

args = parser.parse_args()

# Load edge and node information from CSV
edges_df = pd.read_csv(args.edges, header=None, usecols=[1, 2, 42])
nodes_df = pd.read_csv(args.nodes, header=None, usecols=range(14))

node1_ids = edges_df.iloc[:, 0].values
node2_ids = edges_df.iloc[:, 1].values
edge_weights = edges_df.iloc[:, 2].values

node_ids = nodes_df.index.values
node_features = nodes_df.values

# Standardize node features
scaler = StandardScaler()
node_features = scaler.fit_transform(node_features)

# Create a mapping from node id to node index
node_id_map = {node_id: i for i, node_id in enumerate(node_ids)}

# Identify and print values only in edges, not in node_id_map, if there are any
edges_not_in_node_id_map = {
    'node1_only_in_edges': [id1 for id1 in node1_ids if id1 not in node_id_map],
    'node2_only_in_edges': [id2 for id2 in node2_ids if id2 not in node_id_map]
}
print("Values only in edges, not in node_id_map:", edges_not_in_node_id_map)

# Filter edges to include only those where both nodes exist in the node_id_map
filtered_edges = [(id1, id2) for id1, id2 in zip(node1_ids, node2_ids) if id1 in node_id_map and id2 in node_id_map]
filtered_edge_weights = [edge_weights[i] for i, (id1, id2) in enumerate(zip(node1_ids, node2_ids)) if id1 in node_id_map and id2 in node_id_map]

# Convert to tensors
node_features = torch.tensor(node_features, dtype=torch.float)
edge_index = torch.tensor([[node_id_map[id1], node_id_map[id2]] for id1, id2 in filtered_edges], dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(filtered_edge_weights, dtype=torch.float)

# Create PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# Use RandomLinkSplit to split the dataset for training and testing
transform = RandomLinkSplit(is_undirected=True, num_val=0.1, num_test=0.05, add_negative_train_samples=True)
train_data, val_data, test_data = transform(data)

# Debugging: Print keys to ensure they exist
print("Train data keys:", train_data.keys)
print("Validation data keys:", val_data.keys)
print("Test data keys:", test_data.keys)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(data.num_features, 128, 64).to(device)

# Move data to the device
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    pos_out = model.decode(z, train_data.edge_label_index[:, train_data.edge_label == 1])
    neg_out = model.decode(z, train_data.edge_label_index[:, train_data.edge_label == 0])
    pos_label = torch.ones(pos_out.size(0), device=device)
    neg_label = torch.zeros(neg_out.size(0), device=device)
    out = torch.cat([pos_out, neg_out])
    labels = torch.cat([pos_label, neg_label])
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    return loss

def evaluate(data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
    pos_out = model.decode(z, data.edge_label_index[:, data.edge_label == 1])
    neg_out = model.decode(z, data.edge_label_index[:, data.edge_label == 0])
    pos_label = torch.ones(pos_out.size(0), device=device)
    neg_label = torch.zeros(neg_out.size(0), device=device)
    out = torch.cat([pos_out, neg_out])
    labels = torch.cat([pos_label, neg_label])
    loss = criterion(out, labels)
    
    # Calculate accuracy and f1 score
    preds = torch.sigmoid(out) > 0.5
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    return loss, accuracy, f1

# Training loop with early stopping
best_val_loss = float('inf')
patience = 20
patience_counter = 0

for epoch in range(1, 20001):
    train_loss = train()
    val_loss, val_accuracy, val_f1 = evaluate(val_data)
    test_loss, test_accuracy, test_f1 = evaluate(test_data)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter == patience:
        print(f'Early stopping at epoch {epoch}')
        break
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}')

# Save embeddings
model.eval()
with torch.no_grad():
    z = model.encode(train_data.x, train_data.edge_index).cpu().numpy()

# Save node embeddings to CSV
embeddings_df = pd.DataFrame(z, index=node_ids)
embeddings_df.to_csv(args.output, header=False, float_format='%.12f')
