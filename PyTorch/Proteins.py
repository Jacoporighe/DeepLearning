import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import KFold

# Constants
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AMINO_ACID_DICT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
MAX_SEQ_LEN = 200  # Example fixed length for sequences

def one_hot_encode(sequence, max_len=MAX_SEQ_LEN):
    """One-hot encode the amino acid sequence."""
    encoded = np.zeros((max_len, len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(sequence[:max_len]):
        if aa in AMINO_ACID_DICT:
            encoded[i, AMINO_ACID_DICT[aa]] = 1.0
    return encoded

# Load the dataset
data = scipy.io.loadmat('CancerLectin.mat')
train_sequences = data['Training'][0]  # Unpack the first dimension
train_labels = data['label'].flatten()
test_sequences = data['Test'][0]  # Unpack the first dimension
test_labels = data['labelTEST'].flatten()

# Ensure labels are binary: 1 -> 0.0 and 2 -> 1.0
train_labels = (train_labels == 2).astype(np.float32)
test_labels = (test_labels == 2).astype(np.float32)

# Prepare data
encoded_train_sequences = [one_hot_encode(seq) for seq in train_sequences]
encoded_train_sequences = np.stack(encoded_train_sequences)
encoded_test_sequences = [one_hot_encode(seq) for seq in test_sequences]
encoded_test_sequences = np.stack(encoded_test_sequences)

# Convert to PyTorch tensors
train_data = torch.tensor(encoded_train_sequences).permute(0, 2, 1)  # Permute to (batch_size, channels, seq_len)
train_labels = torch.tensor(train_labels).float()
test_data = torch.tensor(encoded_test_sequences).permute(0, 2, 1)  # Permute to (batch_size, channels, seq_len)
test_labels = torch.tensor(test_labels).float()

# Remove overlapping sequences between training and test sets
train_set = set(map(tuple, train_sequences))
test_set = set(map(tuple, test_sequences))
overlapping_sequences = train_set.intersection(test_set)
train_indices = [i for i, seq in enumerate(train_sequences) if tuple(seq) not in overlapping_sequences]
test_indices = [i for i, seq in enumerate(test_sequences) if tuple(seq) not in overlapping_sequences]

train_data = train_data[train_indices]
train_labels = train_labels[train_indices]
test_data = test_data[test_indices]
test_labels = test_labels[test_indices]

# Create TensorDataset
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

# CNN Model with Dropout
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=len(AMINO_ACIDS), out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)  # Increased dropout rate
        self.fc1 = nn.Linear(32 * (MAX_SEQ_LEN // 2), 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout after first fully connected layer
        x = torch.sigmoid(self.fc2(x)).squeeze(1)
        return x

# Training function with Cross-Validation and Early Stopping
def train_model_kfold(dataset, k=5, num_epochs=20, lr=0.001):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}/{k}')
        
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        model = CNNModel()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

        best_val_loss = float('inf')
        patience, trials = 5, 0  # Early stopping parameters

        for epoch in range(num_epochs):
            model.train()
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    correct += (predicted == batch_labels).sum().item()
                    total += batch_labels.size(0)
            
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trials = 0
            else:
                trials += 1
                if trials >= patience:
                    print("Early stopping triggered")
                    break
        
        fold_results.append(val_accuracy)
    
    print(f'Cross-Validation Results: {fold_results}')
    print(f'Mean Accuracy: {np.mean(fold_results):.4f}, Std Dev: {np.std(fold_results):.4f}')

# Evaluate the Model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            test_outputs = model(batch_data)
            predicted = (test_outputs > 0.5).float()
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
    
    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Perform Cross-Validation
train_model_kfold(train_dataset, k=5, num_epochs=20, lr=0.001)

# Final Evaluation on Test Set
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
final_model = CNNModel()
final_model.train()
# You should train the final model on the entire train_dataset before evaluating
criterion = nn.BCELoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=0.01)
for epoch in range(20):
    for batch_data, batch_labels in DataLoader(train_dataset, batch_size=32, shuffle=True):
        optimizer.zero_grad()
        outputs = final_model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

evaluate_model(final_model, test_loader)