import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score, 
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict

# Load the dataset
file_path = '/Users/mayankraj/Desktop/Thesis Codes /archive/ACI-IoT-2023.csv'

# Load and preprocess data
df = pd.read_csv(file_path)
print("Dataset loaded successfully!")

# Label encoding
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])
classes = label_encoder.classes_

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

# Handle missing data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=features, inplace=True)

# Separate features and target
X = df[features].values
y = df['Label'].values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        x = self.fc(hidden[-1])
        return x

# Initialize model, loss, and optimizer
input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(classes)
model = LSTMModel(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
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

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}")

# Evaluate model
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        test_preds.extend(torch.argmax(outputs, axis=1).numpy())
        test_targets.extend(y_batch.numpy())

# Base Metrics
print("\nBase Metrics:")
print(f"Accuracy: {accuracy_score(test_targets, test_preds):.4f}")
print(f"Precision: {precision_score(test_targets, test_preds, average='weighted', zero_division=1):.4f}")
print(f"Recall: {recall_score(test_targets, test_preds, average='weighted', zero_division=1):.4f}")
print(f"F1-Score: {f1_score(test_targets, test_preds, average='weighted', zero_division=1):.4f}")

# Confusion Matrix
cm = confusion_matrix(test_targets, test_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - Class Wise")
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Class-wise Metrics
print("\nClass-Wise Metrics:")
print(classification_report(test_targets, test_preds, target_names=classes, zero_division=1))

# Category Mapping
class_to_category = {
    'Benign': 'Benign',
    'Ping Sweep': 'Reckon',
    'OS Scan': 'Reckon',
    'Vulnerability Scan': 'Reckon',
    'Port Scan': 'Reckon',
    'ICMP Flood': 'DoS',
    'Slowloris': 'DoS',
    'SYN Flood': 'DoS',
    'UDP Flood': 'DoS',
    'DNS Flood': 'DoS',
    'Dictionary Attack': 'Brute Force',
    'ARP Spoofing': 'Spoofing'
}

# Category-Wise Metrics
category_targets = [class_to_category[classes[t]] for t in test_targets]
category_preds = [class_to_category[classes[p]] for p in test_preds]

print("\nCategory-Wise Metrics:")
category_metrics = defaultdict(dict)

categories = set(class_to_category.values())
for category in categories:
    precision = precision_score(category_targets, category_preds, labels=[category], average='weighted', zero_division=1)
    recall = recall_score(category_targets, category_preds, labels=[category], average='weighted', zero_division=1)
    f1 = f1_score(category_targets, category_preds, labels=[category], average='weighted', zero_division=1)
    acc = np.mean(np.array(category_targets) == np.array(category_preds))
    category_metrics[category]['Precision'] = precision
    category_metrics[category]['Recall'] = recall
    category_metrics[category]['F1'] = f1
    category_metrics[category]['Accuracy'] = acc
    print(f"Category: {category}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

# Plot Category Metrics
plt.figure(figsize=(15, 8))
metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    plt.bar(category_metrics.keys(), [category_metrics[cat][metric] for cat in categories])
    plt.title(f"{metric} for Categories")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
