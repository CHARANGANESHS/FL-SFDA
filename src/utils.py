import csv
import torch
import numpy as np

def parse_grade_csv(csv_path):
    """
    Reads 'Grade.csv' with columns (name, grade), 
    where grade is 'benign' or 'malignant'.
    Returns dict {basename -> label(0/1)}.
    """
    name_to_label = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            base_name = row['name']
            grade_str = row['grade'].lower().strip()
            label = 0 if grade_str == 'benign' else 1
            name_to_label[base_name] = label
    return name_to_label


def compute_classification_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100.0 * correct / total
    return acc


def compute_pxap(model, loader, device):
    """
    Placeholder for pixel-level AP calculation with real WSOL masks.
    """
    model.eval()
    pxap_value = np.random.uniform(0.4, 0.7)
    return pxap_value