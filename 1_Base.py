import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/mayankraj/Desktop/Thesis Codes /archive/ACI-IoT-2023.csv'

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at: {file_path}")
except Exception as e:
    raise RuntimeError(f"Error loading the dataset: {e}")

# Debug: Display column names
print("Columns in dataset:", df.columns.tolist())

# Ensure the 'Label' column exists
if 'Label' not in df.columns:
    raise KeyError("The 'Label' column is missing from the dataset. Please check the file.")

# Display unique categories in the 'Label' column
unique_labels = df['Label'].unique()
print("Unique labels:", unique_labels)

# Check for the existence of the 'Timestamp' column
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %I:%M:%S %p", errors='coerce', dayfirst=True)
else:
    print("Warning: 'Timestamp' column not found. Proceeding without it.")

# Encode labels for multi-class classification
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Feature selection
features = [
    'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

# Check for missing features
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    raise ValueError(f"The following features are missing in the dataset: {missing_features}")

# Handle NaN and infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=features, inplace=True)

# Separate features and target
X = df[features].values
y = df['Label'].values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = self.fc(x)
        return x

# Initialize model, loss function, and optimizer
input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(label_encoder.classes_)

model = LSTMModel(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
train_losses, val_losses, val_accuracies = [], [], []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))

    # Validation loop
    model.eval()
    val_loss = 0
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            val_preds.extend(torch.argmax(outputs, axis=1).numpy())
            val_targets.extend(y_batch.numpy())

    val_losses.append(val_loss / len(val_loader))
    val_accuracy = accuracy_score(val_targets, val_preds)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Evaluate on test data
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        test_preds.extend(torch.argmax(outputs, axis=1).numpy())
        test_targets.extend(y_batch.numpy())

# Calculate test metrics
test_accuracy = accuracy_score(test_targets, test_preds)
test_f1 = f1_score(test_targets, test_preds, average='weighted')
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Print classification report
print(classification_report(test_targets, test_preds, target_names=label_encoder.classes_))

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')
plt.show()
