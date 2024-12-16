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

# FGSM Implementation
def fgsm_attack(model, data, target, epsilon):
    """
    Performs FGSM attack by perturbing the input data.
    """
    data.requires_grad = True  # Enable gradient calculation on the input
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    return perturbed_data.detach()  # Detach to prevent further gradient calculations


# Evaluate FGSM with Different Epsilon Values
epsilons = [0.01, 0.05, 0.1]
results = {}

for epsilon in epsilons:
    print(f"\nEvaluating FGSM with epsilon = {epsilon:.2f}")
    adversarial_preds, adversarial_targets = [], []

    model.eval()
    for X_batch, y_batch in test_loader:
        X_batch_adv = fgsm_attack(model, X_batch.clone(), y_batch.clone(), epsilon)
        outputs = model(X_batch_adv)
        adversarial_preds.extend(torch.argmax(outputs, axis=1).numpy())
        adversarial_targets.extend(y_batch.numpy())

    # Base Metrics for Adversarial Data
    adv_accuracy = accuracy_score(adversarial_targets, adversarial_preds)
    adv_precision = precision_score(adversarial_targets, adversarial_preds, average='weighted', zero_division=1)
    adv_recall = recall_score(adversarial_targets, adversarial_preds, average='weighted', zero_division=1)
    adv_f1 = f1_score(adversarial_targets, adversarial_preds, average='weighted', zero_division=1)
    print(f"Adversarial Base Metrics (epsilon={epsilon}):")
    print(f"  Accuracy: {adv_accuracy:.4f}")
    print(f"  Precision: {adv_precision:.4f}")
    print(f"  Recall: {adv_recall:.4f}")
    print(f"  F1-Score: {adv_f1:.4f}")

    # Confusion Matrix for Adversarial Data
    cm_adv = confusion_matrix(adversarial_targets, adversarial_preds)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_adv, annot=True, fmt='d', cmap="Reds", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - FGSM Adversarial Data (epsilon={epsilon})")
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

    # Class-Wise Metrics for Adversarial Data
    print(f"\nAdversarial Class-Wise Metrics (epsilon={epsilon}):")
    print(classification_report(adversarial_targets, adversarial_preds, target_names=classes, zero_division=1))

    # Category-Wise Metrics for Adversarial Data
    category_adv_preds = [class_to_category[classes[p]] for p in adversarial_preds]
    category_adv_targets = [class_to_category[classes[t]] for t in adversarial_targets]

    print(f"\nAdversarial Category-Wise Metrics (epsilon={epsilon}):")
    adversarial_category_metrics = defaultdict(dict)

    for category in categories:
        adv_precision = precision_score(category_adv_targets, category_adv_preds, labels=[category], average='weighted', zero_division=1)
        adv_recall = recall_score(category_adv_targets, category_adv_preds, labels=[category], average='weighted', zero_division=1)
        adv_f1 = f1_score(category_adv_targets, category_adv_preds, labels=[category], average='weighted', zero_division=1)
        adv_acc = np.mean(np.array(category_adv_targets) == np.array(category_adv_preds))
        adversarial_category_metrics[category]['Precision'] = adv_precision
        adversarial_category_metrics[category]['Recall'] = adv_recall
        adversarial_category_metrics[category]['F1'] = adv_f1
        adversarial_category_metrics[category]['Accuracy'] = adv_acc
        print(f"  Category: {category}")
        print(f"    Accuracy: {adv_acc:.4f}")
        print(f"    Precision: {adv_precision:.4f}")
        print(f"    Recall: {adv_recall:.4f}")
        print(f"    F1-Score: {adv_f1:.4f}")

    # Plot Category Metrics for Adversarial Data
    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.bar(adversarial_category_metrics.keys(), [adversarial_category_metrics[cat][metric] for cat in categories])
        plt.title(f"{metric} for Adversarial Categories (epsilon={epsilon})")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Store results for comparison
    results[epsilon] = {
        'Base Metrics': {
            'Accuracy': adv_accuracy,
            'Precision': adv_precision,
            'Recall': adv_recall,
            'F1': adv_f1,
        },
        'Category Metrics': adversarial_category_metrics,
    }

# Comparison of Metrics Across Epsilons
print("\nComparison of Metrics Across Epsilons:")
comparison_metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
for metric in comparison_metrics:
    print(f"\n{metric} Comparison (Original vs Adversarial):")
    print(f"{'Category':<15}{'Original':<10}{'Epsilon=0.01':<15}{'Epsilon=0.05':<15}{'Epsilon=0.1':<15}")
    for category in categories:
        orig_value = category_metrics[category][metric]
        adv_values = [results[eps]['Category Metrics'][category][metric] for eps in epsilons]
        print(f"{category:<15}{orig_value:<10.4f}{adv_values[0]:<15.4f}{adv_values[1]:<15.4f}{adv_values[2]:<15.4f}")


# PGD Implementation
def pgd_attack(model, data, target, epsilon, alpha, num_iter):
    """
    Performs PGD attack by iteratively perturbing the input data.
    """
    perturbed_data = data.clone().detach()
    perturbed_data.requires_grad = True

    for _ in range(num_iter):
        output = model(perturbed_data)
        loss = nn.CrossEntropyLoss()(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_data.grad.data
        perturbed_data = perturbed_data + alpha * data_grad.sign()
        perturbed_data = torch.max(torch.min(perturbed_data, data + epsilon), data - epsilon)  # Project into epsilon-ball
        perturbed_data = perturbed_data.detach().requires_grad_(True)  # Detach to prevent further gradient accumulation

    return perturbed_data.detach()  # Detach to prevent further gradient calculations

# Evaluate PGD with Different Epsilon Values
pgd_epsilons = [0.01, 0.05, 0.1]
pgd_alpha = 0.01  # Step size for PGD
pgd_num_iter = 10  # Number of iterations
pgd_results = {}

for epsilon in pgd_epsilons:
    print(f"\nEvaluating PGD with epsilon = {epsilon:.2f}")
    pgd_preds, pgd_targets = [], []

    model.eval()
    for X_batch, y_batch in test_loader:
        X_batch_pgd = pgd_attack(model, X_batch.clone(), y_batch.clone(), epsilon, pgd_alpha, pgd_num_iter)
        outputs = model(X_batch_pgd)
        pgd_preds.extend(torch.argmax(outputs, axis=1).numpy())
        pgd_targets.extend(y_batch.numpy())

    # Base Metrics for PGD Adversarial Data
    pgd_accuracy = accuracy_score(pgd_targets, pgd_preds)
    pgd_precision = precision_score(pgd_targets, pgd_preds, average='weighted', zero_division=1)
    pgd_recall = recall_score(pgd_targets, pgd_preds, average='weighted', zero_division=1)
    pgd_f1 = f1_score(pgd_targets, pgd_preds, average='weighted', zero_division=1)
    print(f"PGD Base Metrics (epsilon={epsilon}):")
    print(f"  Accuracy: {pgd_accuracy:.4f}")
    print(f"  Precision: {pgd_precision:.4f}")
    print(f"  Recall: {pgd_recall:.4f}")
    print(f"  F1-Score: {pgd_f1:.4f}")

    # Confusion Matrix for PGD Adversarial Data
    cm_pgd = confusion_matrix(pgd_targets, pgd_preds)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_pgd, annot=True, fmt='d', cmap="Oranges", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - PGD Adversarial Data (epsilon={epsilon})")
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

    # Class-Wise Metrics for PGD Adversarial Data
    print(f"\nPGD Class-Wise Metrics (epsilon={epsilon}):")
    print(classification_report(pgd_targets, pgd_preds, target_names=classes, zero_division=1))

    # Category-Wise Metrics for PGD Adversarial Data
    category_pgd_preds = [class_to_category[classes[p]] for p in pgd_preds]
    category_pgd_targets = [class_to_category[classes[t]] for t in pgd_targets]

    print(f"\nPGD Category-Wise Metrics (epsilon={epsilon}):")
    pgd_category_metrics = defaultdict(dict)

    for category in categories:
        pgd_precision = precision_score(category_pgd_targets, category_pgd_preds, labels=[category], average='weighted', zero_division=1)
        pgd_recall = recall_score(category_pgd_targets, category_pgd_preds, labels=[category], average='weighted', zero_division=1)
        pgd_f1 = f1_score(category_pgd_targets, category_pgd_preds, labels=[category], average='weighted', zero_division=1)
        pgd_acc = np.mean(np.array(category_pgd_targets) == np.array(category_pgd_preds))
        pgd_category_metrics[category]['Precision'] = pgd_precision
        pgd_category_metrics[category]['Recall'] = pgd_recall
        pgd_category_metrics[category]['F1'] = pgd_f1
        pgd_category_metrics[category]['Accuracy'] = pgd_acc
        print(f"  Category: {category}")
        print(f"    Accuracy: {pgd_acc:.4f}")
        print(f"    Precision: {pgd_precision:.4f}")
        print(f"    Recall: {pgd_recall:.4f}")
        print(f"    F1-Score: {pgd_f1:.4f}")

    # Plot Category Metrics for PGD Adversarial Data
    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.bar(pgd_category_metrics.keys(), [pgd_category_metrics[cat][metric] for cat in categories])
        plt.title(f"{metric} for PGD Categories (epsilon={epsilon})")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Store results for comparison
    pgd_results[epsilon] = {
        'Base Metrics': {
            'Accuracy': pgd_accuracy,
            'Precision': pgd_precision,
            'Recall': pgd_recall,
            'F1': pgd_f1,
        },
        'Category Metrics': pgd_category_metrics,
    }

# Comparison of FGSM and PGD Metrics Across Epsilons
print("\nComparison of FGSM and PGD Metrics Across Epsilons:")
comparison_metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
for metric in comparison_metrics:
    print(f"\n{metric} Comparison (Original vs FGSM vs PGD):")
    print(f"{'Category':<15}{'Original':<10}{'FGSM (0.01)':<15}{'FGSM (0.05)':<15}{'FGSM (0.1)':<15}{'PGD (0.01)':<15}{'PGD (0.05)':<15}{'PGD (0.1)':<15}")
    for category in categories:
        orig_value = category_metrics[category][metric]
        fgsm_values = [results[eps]['Category Metrics'][category][metric] for eps in epsilons]
        pgd_values = [pgd_results[eps]['Category Metrics'][category][metric] for eps in pgd_epsilons]
        print(f"{category:<15}{orig_value:<10.4f}{fgsm_values[0]:<15.4f}{fgsm_values[1]:<15.4f}{fgsm_values[2]:<15.4f}{pgd_values[0]:<15.4f}{pgd_values[1]:<15.4f}{pgd_values[2]:<15.4f}")

# Robustness Curve for FGSM and PGD
fgsm_accuracies = []
pgd_accuracies = []

# Collect FGSM accuracies for different epsilons
for epsilon in epsilons:
    fgsm_accuracies.append(results[epsilon]['Base Metrics']['Accuracy'])

# Collect PGD accuracies for different epsilons
for epsilon in pgd_epsilons:
    pgd_accuracies.append(pgd_results[epsilon]['Base Metrics']['Accuracy'])

# Plot Robustness Curves
plt.figure(figsize=(10, 6))
plt.plot(epsilons, fgsm_accuracies, marker='o', label='FGSM')
plt.plot(pgd_epsilons, pgd_accuracies, marker='s', label='PGD')
plt.title('Robustness Curve: FGSM vs PGD')
plt.xlabel('Epsilon (Perturbation Strength)')
plt.ylabel('Accuracy')
plt.xticks(epsilons)
plt.legend()
plt.grid(True)
plt.show()



# CLEVER Score Approximation
def calculate_custom_clever_score(model, data_loader, norm, epsilon=0.1):
    """
    Approximate CLEVER scores for robustness analysis.
    """
    clever_scores = defaultdict(dict)
    for category in categories:
        category_data = [(X, y) for X, y in zip(X_test_tensor, y_test_tensor)
                         if class_to_category[classes[y]] == category]

        if not category_data:
            clever_scores[category][f"CLEVER_L{norm}"] = 0.0
            continue

        category_scores = []
        for X, y in category_data:
            X.requires_grad = True
            output = model(X.unsqueeze(0))
            loss = nn.CrossEntropyLoss()(output, torch.tensor([y]))
            model.zero_grad()
            loss.backward()
            grad = X.grad.data

            if norm == 2:
                grad_norm = torch.norm(grad, p=2)
            elif norm == np.inf:
                grad_norm = torch.max(torch.abs(grad))

            if grad_norm != 0:
                category_scores.append(epsilon / grad_norm.item())

        clever_scores[category][f"CLEVER_L{norm}"] = np.mean(category_scores) if category_scores else 0.0

    return clever_scores

# Perturbability Metric
def calculate_perturbability(model, data_loader, epsilon_values, attack_fn):
    """
    Calculate perturbability as the average drop in accuracy under increasing epsilon values.
    """
    original_accuracy = accuracy_score(test_targets, test_preds)
    perturbability_scores = defaultdict(dict)

    for category in categories:
        print(f"Calculating perturbability for category: {category}")
        category_data = [(X, y) for X, y in zip(X_test_tensor, y_test_tensor)
                         if class_to_category[classes[y]] == category]

        if not category_data:
            perturbability_scores[category]['Perturbability'] = 0.0
            continue

        perturbed_accuracies = []

        for epsilon in epsilon_values:
            adversarial_preds = []
            for X, y in category_data:
                perturbed_X = attack_fn(model, X.unsqueeze(0), torch.tensor([y]), epsilon)
                output = model(perturbed_X)
                pred = torch.argmax(output, axis=1).item()
                adversarial_preds.append(pred)

            # Calculate accuracy drop
            perturbed_accuracy = accuracy_score([y for _, y in category_data], adversarial_preds)
            perturbed_accuracies.append(perturbed_accuracy)

        perturbability_scores[category]['Perturbability'] = original_accuracy - np.mean(perturbed_accuracies)

    return perturbability_scores

# CLEVER Scores for FGSM and PGD
fgsm_clever_scores_l2 = calculate_custom_clever_score(model, test_loader, norm=2)
fgsm_clever_scores_linf = calculate_custom_clever_score(model, test_loader, norm=np.inf)
pgd_clever_scores_l2 = calculate_custom_clever_score(model, test_loader, norm=2)
pgd_clever_scores_linf = calculate_custom_clever_score(model, test_loader, norm=np.inf)

# Perturbability Scores for FGSM and PGD
fgsm_perturbability = calculate_perturbability(model, test_loader, epsilons, fgsm_attack)
pgd_perturbability = calculate_perturbability(model, test_loader, pgd_epsilons,
                                              lambda m, d, t, e: pgd_attack(m, d, t, e, pgd_alpha, pgd_num_iter))

# Display CLEVER and Perturbability Scores
print("\nCustom CLEVER and Perturbability Scores for FGSM:")
for category in categories:
    print(f"Category: {category}")
    print(f"  CLEVER_L2: {fgsm_clever_scores_l2[category]['CLEVER_L2']:.4f}")
    print(f"  CLEVER_Linf: {fgsm_clever_scores_linf[category]['CLEVER_Linf']:.4f}")
    print(f"  Perturbability: {fgsm_perturbability[category]['Perturbability']:.4f}")

print("\nCustom CLEVER and Perturbability Scores for PGD:")
for category in categories:
    print(f"Category: {category}")
    print(f"  CLEVER_L2: {pgd_clever_scores_l2[category]['CLEVER_L2']:.4f}")
    print(f"  CLEVER_Linf: {pgd_clever_scores_linf[category]['CLEVER_Linf']:.4f}")
    print(f"  Perturbability: {pgd_perturbability[category]['Perturbability']:.4f}")

# Convert categories to a sorted list for consistent plotting
categories_list = sorted(list(categories))

# Plot CLEVER and Perturbability Scores
clever_metrics = ['CLEVER_L2', 'CLEVER_Linf']
perturbability_metrics = ['Perturbability']

for metric in clever_metrics + perturbability_metrics:
    plt.figure(figsize=(12, 6))
    if metric in clever_metrics:
        plt.bar(categories_list, [fgsm_clever_scores_l2[cat][metric] if metric in fgsm_clever_scores_l2[cat] else 0
                                  for cat in categories_list], label=f"FGSM {metric}", alpha=0.6)
        plt.bar(categories_list, [pgd_clever_scores_l2[cat][metric] if metric in pgd_clever_scores_l2[cat] else 0
                                  for cat in categories_list], label=f"PGD {metric}", alpha=0.6)
    else:
        plt.bar(categories_list, [fgsm_perturbability[cat][metric] if metric in fgsm_perturbability[cat] else 0
                                  for cat in categories_list], label=f"FGSM {metric}", alpha=0.6)
        plt.bar(categories_list, [pgd_perturbability[cat][metric] if metric in pgd_perturbability[cat] else 0
                                  for cat in categories_list], label=f"PGD {metric}", alpha=0.6)
    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.title(f"Comparison of {metric} for FGSM and PGD")
    plt.legend()
    plt.tight_layout()
    plt.show()
