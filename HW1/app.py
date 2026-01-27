from pathlib import Path
import os, zipfile
from urllib.request import urlretrieve

DATA_DIR = Path("data")

def ensure_data():
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print("✅ data/ already present")
        return

    url = os.environ.get("DATA_ZIP_URL")
    if not url:
        raise RuntimeError("DATA_ZIP_URL not set")

    zip_path = Path("/tmp/data.zip")
    print("⬇️ Downloading data.zip...")
    urlretrieve(url, zip_path)

    print("📦 Extracting data.zip...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")

    if not DATA_DIR.exists():
        raise RuntimeError("❌ data/ not found after extraction")

    print("✅ Data ready")

ensure_data()

import random
import time
import sqlite3

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                            recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Constants
IMG_SIZE = 32
FLATTEN_SIZE = IMG_SIZE * IMG_SIZE  # 1024
SNORLAX_DIR = DATA_DIR / "snorlax"
MUDKIP_DIR = DATA_DIR / "mudkip"
HW1_DIR = Path(__file__).resolve().parent
NON_AI_NOTES_FILE = HW1_DIR / "non_ai_notes.md"


# ============================================================================
# HELPER FUNCTIONS FOR PREPROCESSING
# ============================================================================

def load_image_with_padding(image_path, target_size=32):
    """
    Load an image and resize it to target_size x target_size while preserving
    aspect ratio. Pad with black pixels as needed.
    
    Args:
        image_path: Path to image file
        target_size: Target dimension (default 32)
    
    Returns:
        PIL Image of size (target_size, target_size)
    """
    img = Image.open(image_path).convert('RGB')
    
    # Calculate scaling factor to fit within target size
    width, height = img.size
    scale = target_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create black canvas
    canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    
    # Paste resized image in center
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    canvas.paste(img_resized, (x_offset, y_offset))
    
    return canvas


def preprocess_image(image, augment=False):
    """
    Preprocess a PIL image:
    1. Apply augmentation (optional)
    2. Convert to grayscale
    3. Normalize to [0, 1]
    4. Flatten to 1024-d vector
    
    Args:
        image: PIL Image (32x32 RGB)
        augment: Whether to apply augmentation
    
    Returns:
        1D numpy array of length 1024
    """
    if augment:
        # Apply random augmentations
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random rotation
        angle = random.uniform(-10, 10)
        image = image.rotate(angle, fillcolor=(0, 0, 0))
        
        # Random shift
        shift_x = random.randint(-2, 2)
        shift_y = random.randint(-2, 2)
        image = image.transform(
            image.size,
            Image.AFFINE,
            (1, 0, shift_x, 0, 1, shift_y),
            fillcolor=(0, 0, 0)
        )
    
    # Convert to grayscale
    image_gray = image.convert('L')
    
    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image_gray, dtype=np.float32) / 255.0
    
    # Flatten to 1D vector
    image_flat = image_array.flatten()
    
    return image_flat


# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================

class PokemonDataset(Dataset):
    """
    Custom PyTorch Dataset for Snorlax vs Mudkip classification.
    
    Returns:
        (x, y) where x is a 1024-d float tensor and y is 0 (Snorlax) or 1 (Mudkip)
    """
    
    def __init__(self, image_paths, labels, augment=False):
        """
        Args:
            image_paths: List of paths to image files
            labels: List of labels (0 or 1)
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess image with padding
        image = load_image_with_padding(self.image_paths[idx], target_size=IMG_SIZE)
        
        # Preprocess with optional augmentation
        image_vector = preprocess_image(image, augment=self.augment)
        
        # Convert to tensors
        x = torch.from_numpy(image_vector).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return x, y


def load_data_paths(snorlax_dir, mudkip_dir):
    """
    Load image paths and labels from directories.
    
    Returns:
        image_paths: List of image file paths
        labels: List of labels (0 for Snorlax, 1 for Mudkip)
    """
    image_paths = []
    labels = []
    
    # Load Snorlax images (label 0)
    for img_file in os.listdir(snorlax_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_paths.append(snorlax_dir / img_file)
            labels.append(0)
    
    # Load Mudkip images (label 1)
    for img_file in os.listdir(mudkip_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_paths.append(mudkip_dir / img_file)
            labels.append(1)
    
    return image_paths, labels


# ============================================================================
# TRAIN/VALIDATION SPLIT
# ============================================================================

def train_val_split(image_paths, labels, train_ratio=0.8, seed=SEED):
    """
    Split data into training and validation sets with stratification.
    Ensures both sets have similar class distribution.
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        train_ratio: Ratio of training data (default 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        train_paths, train_labels, val_paths, val_labels
    """
    # Separate by class for stratified split
    class_0_data = [(p, l) for p, l in zip(image_paths, labels) if l == 0]
    class_1_data = [(p, l) for p, l in zip(image_paths, labels) if l == 1]
    
    # Shuffle each class
    random.seed(seed)
    random.shuffle(class_0_data)
    random.shuffle(class_1_data)
    
    # Split each class
    split_0 = int(len(class_0_data) * train_ratio)
    split_1 = int(len(class_1_data) * train_ratio)
    
    train_data = class_0_data[:split_0] + class_1_data[:split_1]
    val_data = class_0_data[split_0:] + class_1_data[split_1:]
    
    # Shuffle combined sets
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    # Unzip
    train_paths, train_labels = zip(*train_data) if train_data else ([], [])
    val_paths, val_labels = zip(*val_data) if val_data else ([], [])
    
    return list(train_paths), list(train_labels), list(val_paths), list(val_labels)


# ============================================================================
# MODEL IMPLEMENTATIONS
# ============================================================================

class LinearModel(nn.Module):
    """
    Linear baseline model (logistic regression).
    Architecture: 1024 -> 2 (no hidden layers)
    """
    
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(FLATTEN_SIZE, 2)
    
    def forward(self, x):
        return self.fc(x)


class MLPModel(nn.Module):
    """
    Configurable Multi-Layer Perceptron.
    
    Args:
        depth: Number of hidden layers (1, 2, or 3)
        width: Number of units per hidden layer
    """
    
    def __init__(self, depth=1, width=128):
        super(MLPModel, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(FLATTEN_SIZE, width))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(width, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPModelWithActivation(nn.Module):
    """
    Multi-Layer Perceptron with configurable output activation.
    
    Args:
        depth: Number of hidden layers (Part A uses 2)
        width: Number of units per hidden layer
        activation: 'sigmoid' or 'step'
    """
    
    def __init__(self, depth=2, width=64, activation='sigmoid'):
        super(MLPModelWithActivation, self).__init__()
        
        self.activation = activation
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(FLATTEN_SIZE, width))
        layers.append(nn.ReLU())
        
        # Hidden layers (depth - 1, so for depth=2 we have 1 hidden layer)
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        
        # Output layer (no activation here - we'll apply it in forward)
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(width, 2)
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Get logits from network
        x = self.network(x)
        logits = self.output_layer(x)
        
        # Apply activation function
        if self.activation == 'sigmoid':
            # Apply sigmoid to each logit, then normalize
            probs = self.sigmoid(logits)
            # Normalize to get valid probabilities
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
            # Return log probabilities for CrossEntropyLoss
            return torch.log(probs + 1e-8)
        else:  # step
            # Use very steep sigmoid to approximate step (still differentiable for training)
            # Temperature of 100 makes it very steep
            step_approx = self.sigmoid(logits * 100.0)
            # Normalize
            probs = step_approx / (step_approx.sum(dim=1, keepdim=True) + 1e-8)
            return torch.log(probs + 1e-8)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=30, learning_rate=0.001, device='cpu'):
    """
    Train a model and track accuracy/loss.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
    
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total if val_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return history, model


# ============================================================================
# PART B.1: DECISION BOUNDARY FUNCTIONS
# ============================================================================

class SimplePerceptron(nn.Module):
    """
    Simple perceptron for 2D binary classification.
    Implements: z = w₁x₁ + w₂x₂ + b, y = g(z)
    """
    def __init__(self):
        super(SimplePerceptron, self).__init__()
        self.linear = nn.Linear(2, 1)  # 2 inputs, 1 output
    
    def forward(self, x):
        return self.linear(x)


def generate_synthetic_data(n_samples=200, noise=0.1, seed=42):
    """
    Generate synthetic 2D data with two classes.
    
    Args:
        n_samples: Number of samples per class
        noise: Amount of noise to add
        seed: Random seed
    
    Returns:
        X: numpy array of shape (2*n_samples, 2)
        y: numpy array of shape (2*n_samples,)
    """
    np.random.seed(seed)
    
    # Class 0: centered around (-1, -1)
    X0 = np.random.randn(n_samples, 2) * noise + np.array([-1, -1])
    y0 = np.zeros(n_samples)
    
    # Class 1: centered around (1, 1)
    X1 = np.random.randn(n_samples, 2) * noise + np.array([1, 1])
    y1 = np.ones(n_samples)
    
    # Combine
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def train_perceptron_2d(X, y, epochs=100, lr=0.01):
    """
    Train a simple perceptron on 2D data.
    
    Args:
        X: numpy array of shape (n_samples, 2)
        y: numpy array of shape (n_samples,)
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        model: Trained PyTorch model
        history: Training history
    """
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    # Create model
    model = SimplePerceptron()
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with sigmoid
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    history = {'loss': []}
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        history['loss'].append(loss.item())
    
    return model, history


def compute_decision_boundary(model, x_range=(-3, 3), n_points=100):
    """
    Compute decision boundary: solve w₁x₁ + w₂x₂ + b = 0 for x₂.
    
    The decision boundary is: x₂ = -(w₁x₁ + b) / w₂
    
    Args:
        model: Trained perceptron model
        x_range: Tuple (min, max) for x₁ range
        n_points: Number of points to compute
    
    Returns:
        x1_line: x₁ values
        x2_line: x₂ values (decision boundary)
    """
    # Get weights and bias
    w = model.linear.weight.data[0].numpy()  # [w₁, w₂]
    b = model.linear.bias.data[0].item()
    
    # Generate x₁ values
    x1_line = np.linspace(x_range[0], x_range[1], n_points)
    
    # Solve w₁x₁ + w₂x₂ + b = 0 for x₂
    # x₂ = -(w₁x₁ + b) / w₂
    if abs(w[1]) > 1e-6:  # Avoid division by zero
        x2_line = -(w[0] * x1_line + b) / w[1]
    else:
        # If w₂ ≈ 0, boundary is vertical
        x2_line = np.full_like(x1_line, -b / w[0] if abs(w[0]) > 1e-6 else 0)
    
    return x1_line, x2_line, w, b


def plot_decision_boundary(X, y, model, title="Decision Boundary Visualization"):
    """
    Plot data points and decision boundary.
    
    Args:
        X: numpy array of shape (n_samples, 2)
        y: numpy array of shape (n_samples,)
        model: Trained perceptron model
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot data points
    class_0 = y == 0
    class_1 = y == 1
    
    ax.scatter(X[class_0, 0], X[class_0, 1], c='blue', s=50, alpha=0.6, 
               label='Class 0', edgecolors='black', linewidth=0.5)
    ax.scatter(X[class_1, 0], X[class_1, 1], c='red', s=50, alpha=0.6, 
               label='Class 1', edgecolors='black', linewidth=0.5)
    
    # Compute and plot decision boundary
    x1_line, x2_line, w, b = compute_decision_boundary(model)
    ax.plot(x1_line, x2_line, 'g-', linewidth=3, label='Decision Boundary', 
            linestyle='--', alpha=0.8)
    
    # Add prediction regions (shaded areas)
    # Create a mesh grid
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                          np.linspace(x2_min, x2_max, 100))
    
    # Predict on mesh grid
    grid_points = torch.FloatTensor(np.c_[xx1.ravel(), xx2.ravel()])
    model.eval()
    with torch.no_grad():
        Z = torch.sigmoid(model(grid_points)).numpy()
    Z = Z.reshape(xx1.shape)
    
    # Plot filled contours for prediction regions
    contour = ax.contourf(xx1, xx2, Z, levels=[0, 0.5, 1], alpha=0.2, 
                         colors=['blue', 'red'], zorder=0)
    
    # Add labels and formatting
    ax.set_xlabel('x₁ (Feature 1)', fontsize=12)
    ax.set_ylabel('x₂ (Feature 2)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add equation text
    w, b = model.linear.weight.data[0].numpy(), model.linear.bias.data[0].item()
    eq_text = f'Boundary: {w[0]:.2f}x₁ + {w[1]:.2f}x₂ + {b:.2f} = 0'
    ax.text(0.02, 0.98, eq_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def visualize_decision_boundary_perceptron_lab_version(w1, w2, bias, X, y):
    """
    EXACT COPY of Perceptron Lab's decision boundary visualization.
    This version has ERRORS that we'll identify and fix.
    
    Errors in this code:
    1. Hardcoded range [0, 10] doesn't adapt to actual data range
    2. Filters boundary line to only show x2 in [0, 10], cutting off the line
    3. If data is outside [0, 10], boundary may not be visible
    4. Doesn't use the actual data range, so boundary might not align with data
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ERROR 1: Hardcoded range [0, 10] - doesn't match actual data!
    x1_range = np.linspace(0, 10, 100)
    x2_range = np.linspace(0, 10, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Compute predictions for grid (using sigmoid activation)
    grid_points = np.c_[X1.ravel(), X2.ravel()]
    Z = 1 / (1 + np.exp(-(w1 * grid_points[:, 0] + w2 * grid_points[:, 1] + bias)))
    Z = Z.reshape(X1.shape)
    
    # Plot decision regions
    colors_dark_light = ['#8B0000', '#2c2c2c', '#1a1a2e', '#16213e', '#0f3460']
    cmap_dark_light = LinearSegmentedColormap.from_list('darklight', colors_dark_light)
    contour = ax.contourf(X1, X2, Z, levels=50, cmap=cmap_dark_light, alpha=0.8)
    
    # Plot data points
    class_0 = y == 0
    class_1 = y == 1
    ax.scatter(X[class_0, 0], X[class_0, 1], c='blue', s=50, alpha=0.6, 
               label='Class 0', edgecolors='black', linewidth=0.5)
    ax.scatter(X[class_1, 0], X[class_1, 1], c='red', s=50, alpha=0.6, 
               label='Class 1', edgecolors='black', linewidth=0.5)
    
    # ERROR 2 & 3: Decision boundary computation with hardcoded range and filtering
    # Decision boundary line: w1*x1 + w2*x2 + b = 0
    # x2 = (-w1*x1 - b) / w2
    if abs(w2) > 0.001:
        x1_line = np.linspace(0, 10, 100)  # ERROR: Hardcoded [0, 10]
        x2_line = (-w1 * x1_line - bias) / w2
        # ERROR: Filters to only show x2 in [0, 10], cutting off the line!
        valid = (x2_line >= 0) & (x2_line <= 10)
        ax.plot(x1_line[valid], x2_line[valid], 'g-', linewidth=3,
               label='Decision Boundary (Perceptron Lab - WITH ERRORS)', linestyle='--')
    
    ax.set_xlabel('x₁ (Feature 1)', fontsize=12)
    ax.set_ylabel('x₂ (Feature 2)', fontsize=12)
    ax.set_title('Perceptron Lab Version - WITH ERRORS\n(Hardcoded range [0,10], boundary may be cut off)', 
                 fontsize=12, fontweight='bold', color='red')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Show actual data range
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    ax.text(0.02, 0.98, f'Data range: x₁∈[{x1_min:.1f}, {x1_max:.1f}], x₂∈[{x2_min:.1f}, {x2_max:.1f}]\n'
                       f'Plot range: x₁∈[0, 10], x₂∈[0, 10] (HARDCODED - ERROR!)',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    return fig


def visualize_decision_boundary_interactive(n_samples=200, noise=0.1, epochs=100, lr=0.01, show_error=True):
    """
    Interactive function for Gradio to generate data, train model, and visualize.
    Shows both the Perceptron Lab version (with errors) and corrected version.
    
    Args:
        n_samples: Number of samples per class
        noise: Amount of noise
        epochs: Training epochs
        lr: Learning rate
        show_error: Whether to show the Perceptron Lab version with errors
    
    Returns:
        matplotlib figure
    """
    # Generate data
    X, y = generate_synthetic_data(n_samples=n_samples, noise=noise)
    
    # Train model
    model, history = train_perceptron_2d(X, y, epochs=epochs, lr=lr)
    
    # Get learned weights
    w = model.linear.weight.data[0].numpy()
    b = model.linear.bias.data[0].item()
    w1, w2, bias = w[0], w[1], b
    
    if show_error:
        # Show side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left: Perceptron Lab version (WITH ERRORS)
        ax1 = axes[0]
        x1_range = np.linspace(0, 10, 100)
        x2_range = np.linspace(0, 10, 100)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        grid_points = np.c_[X1.ravel(), X2.ravel()]
        Z = 1 / (1 + np.exp(-(w1 * grid_points[:, 0] + w2 * grid_points[:, 1] + bias)))
        Z = Z.reshape(X1.shape)
        
        ax1.contourf(X1, X2, Z, levels=50, cmap='RdYlGn', alpha=0.3)
        class_0 = y == 0
        class_1 = y == 1
        ax1.scatter(X[class_0, 0], X[class_0, 1], c='blue', s=50, alpha=0.6, 
                   label='Class 0', edgecolors='black', linewidth=0.5)
        ax1.scatter(X[class_1, 0], X[class_1, 1], c='red', s=50, alpha=0.6, 
                   label='Class 1', edgecolors='black', linewidth=0.5)
        
        # ERROR: Hardcoded range and filtering
        if abs(w2) > 0.001:
            x1_line = np.linspace(0, 10, 100)
            x2_line = (-w1 * x1_line - bias) / w2
            valid = (x2_line >= 0) & (x2_line <= 10)
            ax1.plot(x1_line[valid], x2_line[valid], 'g--', linewidth=3,
                   label='Boundary (CUT OFF!)', alpha=0.7)
        
        ax1.set_xlabel('x₁', fontsize=12)
        ax1.set_ylabel('x₂', fontsize=12)
        ax1.set_title('❌ Perceptron Lab Version (WITH ERRORS)\nHardcoded [0,10] range, boundary cut off',
                     fontsize=12, fontweight='bold', color='red')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        
        # Right: Corrected version
        ax2 = axes[1]
        plot_decision_boundary_on_axes(X, y, model, ax2, title='✅ Corrected Version\nAdaptive range, full boundary')
        
        plt.suptitle('Comparison: Perceptron Lab Code vs Corrected Implementation', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
    else:
        # Just show corrected version
        fig = plot_decision_boundary(X, y, model, 
                                    title="Decision Boundary: Perceptron Classification")
    
    return fig


def plot_decision_boundary_on_axes(X, y, model, ax, title="Decision Boundary"):
    """Helper to plot on existing axes."""
    class_0 = y == 0
    class_1 = y == 1
    
    ax.scatter(X[class_0, 0], X[class_0, 1], c='blue', s=50, alpha=0.6, 
               label='Class 0', edgecolors='black', linewidth=0.5)
    ax.scatter(X[class_1, 0], X[class_1, 1], c='red', s=50, alpha=0.6, 
               label='Class 1', edgecolors='black', linewidth=0.5)
    
    # Compute decision boundary with adaptive range
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x1_line = np.linspace(x1_min, x1_max, 100)
    w = model.linear.weight.data[0].numpy()
    b = model.linear.bias.data[0].item()
    
    if abs(w[1]) > 1e-6:
        x2_line = -(w[0] * x1_line + b) / w[1]
        ax.plot(x1_line, x2_line, 'g-', linewidth=3, label='Decision Boundary', 
                linestyle='--', alpha=0.8)
    
    # Prediction regions
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                          np.linspace(x2_min, x2_max, 100))
    grid_points = torch.FloatTensor(np.c_[xx1.ravel(), xx2.ravel()])
    model.eval()
    with torch.no_grad():
        Z = torch.sigmoid(model(grid_points)).numpy()
    Z = Z.reshape(xx1.shape)
    ax.contourf(xx1, xx2, Z, levels=[0, 0.5, 1], alpha=0.2, 
               colors=['blue', 'red'], zorder=0)
    
    ax.set_xlabel('x₁ (Feature 1)', fontsize=12)
    ax.set_ylabel('x₂ (Feature 2)', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_training_curves(history):
    """
    Plot training and validation accuracy curves.
    
    Args:
        history: Dictionary with training history
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    ax.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_training_curves_compare(history_sigmoid, history_step):
    """
    Plot training and validation accuracy for both models (Sigmoid vs Step) for comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history_sigmoid['train_acc']) + 1)

    ax.plot(epochs, history_sigmoid['train_acc'], 'b-', label='Sigmoid — Train', linewidth=1.5)
    ax.plot(epochs, history_sigmoid['val_acc'], 'b--', label='Sigmoid — Val', linewidth=1.5)
    ax.plot(epochs, history_step['train_acc'], 'orange', linestyle='-', label='Step — Train', linewidth=1.5)
    ax.plot(epochs, history_step['val_acc'], 'orange', linestyle='--', label='Step — Val', linewidth=1.5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Part A: Training vs Validation Accuracy (Sigmoid vs Step)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================

def train_and_evaluate(activation, augmentation):
    """
    Main function called by Gradio interface to train and evaluate models.
    
    For Part A: Compare sigmoid vs step activation functions.
    Fixed: 2 hidden layers, width=64, epochs=30, lr=0.001
    
    Returns:
        results_text: String with training results
        plot: matplotlib figure with training curves
        cm_fig: matplotlib figure with confusion matrix
    """
    start_time = time.time()
    
    # Fixed hyperparameters for Part A
    FIXED_DEPTH = 2  # 2 hidden layers
    FIXED_WIDTH = 64
    FIXED_EPOCHS = 30
    FIXED_LR = 0.001
    
    # Load data
    image_paths, labels = load_data_paths(SNORLAX_DIR, MUDKIP_DIR)
    
    # Split data
    train_paths, train_labels, val_paths, val_labels = train_val_split(
        image_paths, labels, train_ratio=0.8
    )
    
    # Create datasets
    train_dataset = PokemonDataset(train_paths, train_labels, augment=augmentation)
    val_dataset = PokemonDataset(val_paths, val_labels, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model with specified activation
    model = MLPModelWithActivation(depth=FIXED_DEPTH, width=FIXED_WIDTH, activation=activation)
    activation_name = "Sigmoid" if activation == "sigmoid" else "Step"
    model_name = f"Neural Network (2 layers, {activation_name} activation, width={FIXED_WIDTH})"
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    history, trained_model = train_model(
        model, train_loader, val_loader,
        epochs=FIXED_EPOCHS, learning_rate=FIXED_LR, device=device
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Get final metrics
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    
    # Get predictions on validation set for confusion matrix
    trained_model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = trained_model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(labels.numpy())
    
    # Count class distribution
    train_snorlax = sum(1 for l in train_labels if l == 0)
    train_mudkip = sum(1 for l in train_labels if l == 1)
    val_snorlax = sum(1 for l in val_labels if l == 0)
    val_mudkip = sum(1 for l in val_labels if l == 1)
    
    # Create results text with clean markdown formatting
    results_text = f"""
### TRAINING RESULTS

**Model:** {model_name}

**Fixed Hyperparameters:**
- Network Depth: {FIXED_DEPTH} hidden layers
- Hidden Layer Width: {FIXED_WIDTH}
- Epochs: {FIXED_EPOCHS}
- Learning Rate: {FIXED_LR}
- Activation Function: {activation_name}
- Augmentation: {'Enabled' if augmentation else 'Disabled'}

**Training Set Size:** {len(train_paths)} images
- Snorlax: {train_snorlax}
- Mudkip: {train_mudkip}

**Validation Set Size:** {len(val_paths)} images
- Snorlax: {val_snorlax}
- Mudkip: {val_mudkip}

**Final Training Accuracy:** {final_train_acc*100:.2f}%

**Final Validation Accuracy:** {final_val_acc*100:.2f}%

**Training Time:** {training_time:.2f} seconds
"""
    
    # Create plots
    plot_fig = plot_training_curves(history)
    cm_fig = plot_confusion_matrix(val_true, val_preds, ["Snorlax", "Mudkip"])

    # Store trained model globally for prediction
    global TRAINED_MODEL, CLASS_NAMES
    TRAINED_MODEL = trained_model
    CLASS_NAMES = ["Snorlax", "Mudkip"]

    return results_text, plot_fig, cm_fig


def train_both_models_part_a(augmentation):
    """
    Train both Sigmoid and Step activation models simultaneously for Part A.
    Fixed: 2 hidden layers, width=64, epochs=30, lr=0.001. Snorlax vs Mudkip.

    Returns:
        results_text: Combined results for both models
        plot_fig: Training vs validation accuracy comparison (both models)
        cm_sigmoid_fig: Confusion matrix for Sigmoid model
        cm_step_fig: Confusion matrix for Step model
    """
    start_time = time.time()
    FIXED_DEPTH = 2
    FIXED_WIDTH = 64
    FIXED_EPOCHS = 30
    FIXED_LR = 0.001

    image_paths, labels = load_data_paths(SNORLAX_DIR, MUDKIP_DIR)
    train_paths, train_labels, val_paths, val_labels = train_val_split(image_paths, labels, train_ratio=0.8)

    train_dataset = PokemonDataset(train_paths, train_labels, augment=augmentation)
    val_dataset = PokemonDataset(val_paths, val_labels, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Train Sigmoid model ---
    model_sigmoid = MLPModelWithActivation(depth=FIXED_DEPTH, width=FIXED_WIDTH, activation='sigmoid')
    history_sigmoid, trained_sigmoid = train_model(
        model_sigmoid, train_loader, val_loader,
        epochs=FIXED_EPOCHS, learning_rate=FIXED_LR, device=device
    )
    trained_sigmoid.eval()
    val_preds_sigmoid = []
    val_true_list = []
    with torch.no_grad():
        for inputs, labels_batch in val_loader:
            inputs = inputs.to(device)
            outputs = trained_sigmoid(inputs)
            _, predicted = torch.max(outputs, 1)
            val_preds_sigmoid.extend(predicted.cpu().numpy())
            val_true_list.extend(labels_batch.numpy())

    # --- Train Step model ---
    model_step = MLPModelWithActivation(depth=FIXED_DEPTH, width=FIXED_WIDTH, activation='step')
    history_step, trained_step = train_model(
        model_step, train_loader, val_loader,
        epochs=FIXED_EPOCHS, learning_rate=FIXED_LR, device=device
    )
    trained_step.eval()
    val_preds_step = []
    with torch.no_grad():
        for inputs, labels_batch in val_loader:
            inputs = inputs.to(device)
            outputs = trained_step(inputs)
            _, predicted = torch.max(outputs, 1)
            val_preds_step.extend(predicted.cpu().numpy())

    elapsed = time.time() - start_time
    train_snorlax = sum(1 for l in train_labels if l == 0)
    train_mudkip = sum(1 for l in train_labels if l == 1)
    val_snorlax = sum(1 for l in val_labels if l == 0)
    val_mudkip = sum(1 for l in val_labels if l == 1)
    class_names = ["Snorlax", "Mudkip"]

    # Build DataFrame for Gradio Dataframe visualization
    results_df = pd.DataFrame([
        {
            "Model": "Sigmoid",
            "Final train acc": f"{history_sigmoid['train_acc'][-1] * 100:.2f}%",
            "Final val acc": f"{history_sigmoid['val_acc'][-1] * 100:.2f}%",
        },
        {
            "Model": "Step",
            "Final train acc": f"{history_step['train_acc'][-1] * 100:.2f}%",
            "Final val acc": f"{history_step['val_acc'][-1] * 100:.2f}%",
        },
    ])

    plot_fig = plot_training_curves_compare(history_sigmoid, history_step)
    cm_sigmoid_fig = plot_confusion_matrix(val_true_list, val_preds_sigmoid, class_names)
    cm_step_fig = plot_confusion_matrix(val_true_list, val_preds_step, class_names)

    global TRAINED_MODEL_SIGMOID, TRAINED_MODEL_STEP, CLASS_NAMES
    TRAINED_MODEL_SIGMOID = trained_sigmoid
    TRAINED_MODEL_STEP = trained_step
    CLASS_NAMES = class_names

    return results_df, plot_fig, cm_sigmoid_fig, cm_step_fig


def predict_image(image):
    """
    Predict the class of an uploaded image.
    
    Args:
        image: PIL Image uploaded by user
    
    Returns:
        String with prediction and confidence
    """
    global TRAINED_MODEL, CLASS_NAMES
    
    if TRAINED_MODEL is None:
        return "Please train a model first before making predictions!"
    
    if image is None:
        return "Please upload an image first!"
    
    # Preprocess the image (image is already a PIL Image from Gradio)
    # Convert to RGB if needed
    img = image.convert('RGB')
    
    # Resize with padding to preserve aspect ratio
    target_size = IMG_SIZE
    width, height = img.size
    scale = target_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create black canvas and paste resized image in center
    canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    canvas.paste(img_resized, (x_offset, y_offset))
    
    # Now preprocess (no augmentation for prediction)
    image_vector = preprocess_image(canvas, augment=False)
    
    # Convert to tensor
    x = torch.from_numpy(image_vector).float().unsqueeze(0)
    
    # Make prediction
    TRAINED_MODEL.eval()
    with torch.no_grad():
        outputs = TRAINED_MODEL(x)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item() * 100
    
    result = f"""
    Prediction: {predicted_class}
    Confidence: {confidence_score:.2f}%
    
    Class Probabilities:
    - Snorlax: {probabilities[0][0].item()*100:.2f}%
    - Mudkip: {probabilities[0][1].item()*100:.2f}%
    """
    
    return result


def predict_image_both(image):
    """
    Run the uploaded image through both Part A models (Sigmoid and Step) and return
    each model's prediction for Snorlax vs Mudkip.
    """
    global TRAINED_MODEL_SIGMOID, TRAINED_MODEL_STEP, CLASS_NAMES

    if TRAINED_MODEL_SIGMOID is None or TRAINED_MODEL_STEP is None:
        return "Please train both models first (click **Train Both Models** above)."

    if image is None:
        return "Please upload an image first."

    img = image.convert('RGB')
    target_size = IMG_SIZE
    width, height = img.size
    scale = target_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    canvas.paste(img_resized, (x_offset, y_offset))
    image_vector = preprocess_image(canvas, augment=False)
    x = torch.from_numpy(image_vector).float().unsqueeze(0)

    def run_model(model):
        model.eval()
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)
        return CLASS_NAMES[pred.item()], conf.item() * 100, probs[0][0].item() * 100, probs[0][1].item() * 100

    s_class, s_conf, s_snorlax, s_mudkip = run_model(TRAINED_MODEL_SIGMOID)
    t_class, t_conf, t_snorlax, t_mudkip = run_model(TRAINED_MODEL_STEP)

    return f"""
### Sigmoid model
- **Prediction:** {s_class}
- **Confidence:** {s_conf:.2f}%
- Snorlax: {s_snorlax:.2f}% | Mudkip: {s_mudkip:.2f}%

### Step model
- **Prediction:** {t_class}
- **Confidence:** {t_conf:.2f}%
- Snorlax: {t_snorlax:.2f}% | Mudkip: {t_mudkip:.2f}%
"""


# Global variables for trained model
TRAINED_MODEL = None
TRAINED_MODEL_SIGMOID = None
TRAINED_MODEL_STEP = None
CLASS_NAMES = ["Snorlax", "Mudkip"]


def load_pixel_snorlax():
    """
    Load and prepare the 32×32 pixel Snorlax image for Part B.0.
    
    Returns:
        PIL Image of size (32, 32) or None if not found
    """
    try:
        snorlax_pixel_path = DATA_DIR / "snorlax_pixelated" / "143.png"
        if snorlax_pixel_path.exists():
            img = Image.open(snorlax_pixel_path)
            # Ensure it's exactly 32x32
            img = img.resize((32, 32), Image.LANCZOS)
            img = img.convert('RGB')
            return img
        else:
            return None
    except Exception as e:
        print(f"Error loading pixel Snorlax: {e}")
        return None


def load_original_snorlax():
    """
    Load the original 143.png file (unmodified, at original size) for comparison.
    
    Returns:
        PIL Image (original size) or None if not found
    """
    try:
        original_path = DATA_DIR / "snorlax_pixelated" / "143.png"
        if original_path.exists():
            img = Image.open(original_path)
            img = img.convert('RGB')
            return img
        else:
            return None
    except Exception as e:
        print(f"Error loading original 143.png: {e}")
        return None


def visualize_pixel_character(img):
    """
    Create visualizations for Part B.0 showing RGB, grayscale, and flattened vector.
    
    Args:
        img: PIL Image (32×32 RGB)
    
    Returns:
        matplotlib figure with multiple subplots
    """
    if img is None:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original RGB image (upscaled for visibility)
    img_rgb_large = img.resize((128, 128), Image.NEAREST)  # Use nearest neighbor to show pixels
    axes[0, 0].imshow(img_rgb_large)
    axes[0, 0].set_title('32×32 Pixel Snorlax (RGB, upscaled 4×)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Grayscale version
    img_gray = img.convert('L')
    img_gray_large = img_gray.resize((128, 128), Image.NEAREST)
    axes[0, 1].imshow(img_gray_large, cmap='gray')
    axes[0, 1].set_title('32×32 Grayscale (for ML input)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Flattened vector visualization (first 100 values)
    img_array = np.array(img_gray, dtype=np.float32) / 255.0
    flattened = img_array.flatten()
    
    axes[1, 0].plot(flattened[:100], 'b-', linewidth=1.5)
    axes[1, 0].set_title(f'Flattened Vector (first 100 of {len(flattened)} values)', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Pixel Index', fontsize=10)
    axes[1, 0].set_ylabel('Normalized Intensity [0, 1]', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Heatmap of pixel values
    im = axes[1, 1].imshow(img_array, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('32×32 Pixel Intensity Matrix', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Width (32 pixels)', fontsize=10)
    axes[1, 1].set_ylabel('Height (32 pixels)', fontsize=10)
    plt.colorbar(im, ax=axes[1, 1], label='Intensity')
    
    plt.suptitle('Pixel Character: From Image to Feature Vector', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


# ============================================================================
# PART B.3: STAR WARS DEATH PREDICTION
# ============================================================================

STAR_WARS_DB = DATA_DIR / "starwars" / "star_wars.db"


def get_star_wars_categorical_options():
    """Load distinct values for each categorical column from the DB for dropdown presets."""
    conn = sqlite3.connect(STAR_WARS_DB)
    df = pd.read_sql_query("SELECT * FROM characters", conn)
    conn.close()
    options = {}
    for col in ['gender', 'species', 'hair_color', 'eye_color', 'skin_color', 'homeworld']:
        if col not in df.columns:
            options[col] = ['Unknown']
            continue
        vals = df[col].dropna().astype(str).str.strip()
        vals = sorted(vals[vals != ''].unique().tolist())
        if vals and 'Unknown' not in vals:
            vals = ['Unknown'] + vals
        elif not vals:
            vals = ['Unknown']
        options[col] = vals
    return options


def load_star_wars_data():
    """
    Load Star Wars character data and prepare for death prediction.
    
    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (binary: 1 if died, 0 if alive)
        feature_names: List of feature names
        scaler: Fitted StandardScaler
        label_encoders: Dictionary of label encoders for categorical features
    """
    conn = sqlite3.connect(STAR_WARS_DB)
    df = pd.read_sql_query("SELECT * FROM characters", conn)
    conn.close()
    
    # Create target: 1 if died (year_died is not null), 0 if alive
    df['died'] = df['year_died'].notna().astype(int)
    
    # Select physical attributes as features
    # Numerical: height, weight, year_born
    # Categorical: gender, species, hair_color, eye_color, skin_color, homeworld (if present)
    feature_cols = ['height', 'weight', 'year_born', 'gender', 'species',
                    'hair_color', 'eye_color', 'skin_color']
    if 'homeworld' in df.columns:
        feature_cols.append('homeworld')
    
    categorical_cols = [c for c in ['gender', 'species', 'hair_color', 'eye_color', 'skin_color', 'homeworld']
                       if c in df.columns]
    
    # Filter to rows with at least height and weight (essential features)
    df_clean = df[df['height'].notna() & df['weight'].notna()].copy()
    
    # Fill missing year_born with median
    df_clean['year_born'] = df_clean['year_born'].fillna(df_clean['year_born'].median())
    
    # Fill missing categoricals with 'Unknown'
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Encode categorical variables
    label_encoders = {}
    X_encoded = df_clean[['height', 'weight', 'year_born']].copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
    
    # Get feature names
    feature_names = list(X_encoded.columns)
    
    # Convert to numpy
    X = X_encoded.values.astype(np.float32)
    y = df_clean['died'].values.astype(np.float32)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, feature_names, scaler, label_encoders, df_clean


class StarWarsDataset(Dataset):
    """Dataset for Star Wars character death prediction."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LinearDeathPredictor(nn.Module):
    """Linear regression (logistic regression) for death prediction."""
    def __init__(self, n_features):
        super(LinearDeathPredictor, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return self.linear(x)


class NeuralNetworkDeathPredictor(nn.Module):
    """Feed-forward neural network for death prediction."""
    def __init__(self, n_features, hidden_size=64):
        super(NeuralNetworkDeathPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        return self.network(x)


def train_death_predictor(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """
    Train a death prediction model.
    
    Returns:
        history: Dictionary with training history
        trained_model: Trained model
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with sigmoid
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        
        val_loss = val_loss / val_total if val_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    
    return history, model


def evaluate_model(model, X, y, device='cpu'):
    """
    Evaluate model and return comprehensive metrics.
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds, zero_division=0),
        'recall': recall_score(y, preds, zero_division=0),
        'f1': f1_score(y, preds, zero_division=0),
        'roc_auc': roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.0,
        'predictions': preds,
        'probabilities': probs
    }
    
    return metrics


def plot_model_comparison(linear_metrics, nn_metrics, linear_history, nn_history, y_test):
    """Create comparison plots for linear vs neural network (3x2: accuracy, loss, metrics, ROC, two confusion matrices)."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    
    epochs = range(1, len(linear_history['train_acc']) + 1)
    
    # Row 0: Accuracy curves
    axes[0, 0].plot(epochs, linear_history['train_acc'], 'b-', label='Linear (Train)', linewidth=2)
    axes[0, 0].plot(epochs, linear_history['val_acc'], 'b--', label='Linear (Val)', linewidth=2)
    axes[0, 0].plot(epochs, nn_history['train_acc'], 'r-', label='Neural Net (Train)', linewidth=2)
    axes[0, 0].plot(epochs, nn_history['val_acc'], 'r--', label='Neural Net (Val)', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('Training Curves: Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Row 0: Loss curves
    axes[0, 1].plot(epochs, linear_history['train_loss'], 'b-', label='Linear (Train)', linewidth=2)
    axes[0, 1].plot(epochs, linear_history['val_loss'], 'b--', label='Linear (Val)', linewidth=2)
    axes[0, 1].plot(epochs, nn_history['train_loss'], 'r-', label='Neural Net (Train)', linewidth=2)
    axes[0, 1].plot(epochs, nn_history['val_loss'], 'r--', label='Neural Net (Val)', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title('Training Curves: Loss', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 1: Metrics comparison
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    linear_vals = [linear_metrics['accuracy'], linear_metrics['precision'],
                   linear_metrics['recall'], linear_metrics['f1'], linear_metrics['roc_auc']]
    nn_vals = [nn_metrics['accuracy'], nn_metrics['precision'],
               nn_metrics['recall'], nn_metrics['f1'], nn_metrics['roc_auc']]
    x = np.arange(len(metrics_names))
    width = 0.35
    axes[1, 0].bar(x - width/2, linear_vals, width, label='Linear Regression', alpha=0.8)
    axes[1, 0].bar(x + width/2, nn_vals, width, label='Neural Network', alpha=0.8)
    axes[1, 0].set_ylabel('Score', fontsize=11)
    axes[1, 0].set_title('Model Comparison: Metrics', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics_names, rotation=45, ha='right')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1.1])
    
    # Row 1: ROC curves
    if len(np.unique(y_test)) > 1:
        fpr_linear, tpr_linear, _ = roc_curve(y_test, linear_metrics['probabilities'])
        fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_metrics['probabilities'])
        axes[1, 1].plot(fpr_linear, tpr_linear, 'b-', label=f'Linear (AUC={linear_metrics["roc_auc"]:.3f})', linewidth=2)
        axes[1, 1].plot(fpr_nn, tpr_nn, 'r-', label=f'Neural Net (AUC={nn_metrics["roc_auc"]:.3f})', linewidth=2)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        axes[1, 1].set_xlabel('False Positive Rate', fontsize=11)
        axes[1, 1].set_ylabel('True Positive Rate', fontsize=11)
        axes[1, 1].set_title('ROC Curves', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'ROC Curves\n(Not enough class diversity)',
                        ha='center', va='center', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('ROC Curves', fontsize=12, fontweight='bold')
    
    # Row 2: Confusion matrix linear
    cm_linear = confusion_matrix(y_test, linear_metrics['predictions'])
    im0 = axes[2, 0].imshow(cm_linear, cmap='Blues', alpha=0.8)
    axes[2, 0].set_title('Confusion Matrix: Linear Regression', fontsize=12, fontweight='bold')
    axes[2, 0].set_xticks([0, 1])
    axes[2, 0].set_yticks([0, 1])
    axes[2, 0].set_xticklabels(['Alive', 'Died'])
    axes[2, 0].set_yticklabels(['Alive', 'Died'])
    axes[2, 0].set_ylabel('True Label', fontsize=10)
    axes[2, 0].set_xlabel('Predicted Label', fontsize=10)
    thresh_linear = cm_linear.max() / 2.
    for i in range(2):
        for j in range(2):
            axes[2, 0].text(j, i, str(cm_linear[i, j]), ha="center", va="center",
                            color="white" if cm_linear[i, j] > thresh_linear else "black", fontsize=11, fontweight='bold')
    
    # Row 2: Confusion matrix neural network
    cm_nn = confusion_matrix(y_test, nn_metrics['predictions'])
    im1 = axes[2, 1].imshow(cm_nn, cmap='Blues', alpha=0.8)
    axes[2, 1].set_title('Confusion Matrix: Neural Network', fontsize=12, fontweight='bold')
    axes[2, 1].set_xticks([0, 1])
    axes[2, 1].set_yticks([0, 1])
    axes[2, 1].set_xticklabels(['Alive', 'Died'])
    axes[2, 1].set_yticklabels(['Alive', 'Died'])
    axes[2, 1].set_ylabel('True Label', fontsize=10)
    axes[2, 1].set_xlabel('Predicted Label', fontsize=10)
    thresh_nn = cm_nn.max() / 2.
    for i in range(2):
        for j in range(2):
            axes[2, 1].text(j, i, str(cm_nn[i, j]), ha="center", va="center",
                            color="white" if cm_nn[i, j] > thresh_nn else "black", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def train_and_compare_models():
    """
    Main function to load data, train both models, and return comparison.
    
    Returns:
        results_text: String with results
        comparison_plot: matplotlib figure
    """
    # Load data
    X, y, feature_names, scaler, label_encoders, df_clean = load_star_wars_data()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create datasets
    train_dataset = StarWarsDataset(X_train, y_train)
    test_dataset = StarWarsDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Train Linear Model
    linear_model = LinearDeathPredictor(n_features=X.shape[1])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    linear_history, linear_trained = train_death_predictor(
        linear_model, train_loader, test_loader, epochs=100, lr=0.001, device=device
    )
    
    # Train Neural Network
    nn_model = NeuralNetworkDeathPredictor(n_features=X.shape[1], hidden_size=64)
    nn_history, nn_trained = train_death_predictor(
        nn_model, train_loader, test_loader, epochs=100, lr=0.001, device=device
    )
    
    # Evaluate both models
    linear_metrics = evaluate_model(linear_trained, X_test, y_test, device=device)
    nn_metrics = evaluate_model(nn_trained, X_test, y_test, device=device)
    
    # Create results text
    results_text = f"""
### Model Comparison Results

**Dataset Information:**
- Total characters: {len(df_clean)}
- Features: {', '.join(feature_names)}
- Died: {int(y.sum())} ({y.sum()/len(y)*100:.1f}%)
- Alive: {int((1-y).sum())} ({(1-y).sum()/len(y)*100:.1f}%)

**Model Specifications:**
- **Linear Regression**: Single layer, logistic regression
- **Neural Network**: 3-layer feed-forward (64 → 32 → 1), ReLU activations, Dropout (0.2)
- **Loss Function**: Binary Cross-Entropy (BCEWithLogitsLoss)
- **Optimizer**: Adam (lr=0.001)
- **Train/Test Split**: 80/20 (stratified)

**Linear Regression Results:**
- Accuracy: {linear_metrics['accuracy']:.3f}
- Precision: {linear_metrics['precision']:.3f}
- Recall: {linear_metrics['recall']:.3f}
- F1 Score: {linear_metrics['f1']:.3f}
- ROC-AUC: {linear_metrics['roc_auc']:.3f}

**Neural Network Results:**
- Accuracy: {nn_metrics['accuracy']:.3f}
- Precision: {nn_metrics['precision']:.3f}
- Recall: {nn_metrics['recall']:.3f}
- F1 Score: {nn_metrics['f1']:.3f}
- ROC-AUC: {nn_metrics['roc_auc']:.3f}

**Conclusion:**
{'Neural Network performs better' if nn_metrics['accuracy'] > linear_metrics['accuracy'] else 'Linear Regression performs better'} on this task.
"""
    
    # Create comparison plot
    comparison_plot = plot_model_comparison(linear_metrics, nn_metrics, linear_history, nn_history, y_test)
    
    # State for interactive predictor: use the better model by accuracy
    best_model = nn_trained if nn_metrics['accuracy'] >= linear_metrics['accuracy'] else linear_trained
    predictor_state = (best_model, scaler, label_encoders, feature_names, device)
    
    return results_text, comparison_plot, predictor_state


def predict_character(state, height, weight, year_born, gender, species, hair_color, eye_color, skin_color, homeworld):
    """
    Predict whether a custom character would die using the trained model.
    Unseen categorical values are mapped to 'Unknown' (or first class) before encoding.
    """
    if state is None:
        return "**Please train models first.** Click \"Train Models & Compare\" above, then try again."
    
    model, scaler, label_encoders, feature_names, device = state
    model.eval()
    
    name_to_val = {
        'height': height, 'weight': weight, 'year_born': year_born,
        'gender': gender, 'species': species, 'hair_color': hair_color,
        'eye_color': eye_color, 'skin_color': skin_color, 'homeworld': homeworld
    }
    
    row = []
    for name in feature_names:
        v = name_to_val.get(name)
        if name in label_encoders:
            v = str(v).strip() if v is not None and str(v).strip() else 'Unknown'
            classes = label_encoders[name].classes_
            if v not in classes:
                v = 'Unknown' if 'Unknown' in classes else classes[0]
            row.append(label_encoders[name].transform([v])[0])
        else:
            try:
                row.append(float(v) if v is not None and str(v).strip() != '' else 0.0)
            except (TypeError, ValueError):
                row.append(0.0)
    
    X_row = np.array([row], dtype=np.float32)
    X_scaled = scaler.transform(X_row)
    x_tensor = torch.FloatTensor(X_scaled).to(device)
    
    with torch.no_grad():
        logit = model(x_tensor).squeeze().item()
    prob = 1.0 / (1.0 + np.exp(-logit))
    pred = "Died" if prob > 0.5 else "Alive"
    
    return f"**Prediction: {pred}**  \nP(died) = {prob:.3f}"


# ============================================================================
# GRADIO APP
# ============================================================================

def create_gradio_interface():
    """
    Create and configure the Gradio interface with tabs for each homework part.
    """
    with gr.Blocks(title="Homework 1 - Applied Machine Learning") as demo:
        gr.Markdown("""
        # STAT UN 3106: Applied Machine Learning - Homework 1
        
        **Jalen Stephens** · js5987 · 02/3/26
        """)
        
        with gr.Tabs() as tabs:
            # ============================================================
            # PREFACE: NON-AI NOTES (for instructor/grader)
            # ============================================================
            with gr.Tab("Preface: Non-AI Notes"):
                def load_non_ai_notes():
                    """Load and return the contents of non_ai_notes.md for display."""
                    if not NON_AI_NOTES_FILE.exists():
                        return (
                            "*No file found.*\n\n"
                            f"Create a file named **`non_ai_notes.md`** in the HW1 folder:\n\n"
                            f"`{NON_AI_NOTES_FILE}`\n\n"
                            "Add your notes and instructions for each part, then click **Refresh** to see it here."
                        )
                    try:
                        with open(NON_AI_NOTES_FILE, "r", encoding="utf-8") as f:
                            return f.read()
                    except Exception as e:
                        return f"*Error reading file: {e}*"
                
                notes_display = gr.Markdown(value=load_non_ai_notes(), label="Contents of non_ai_notes.md")
                refresh_btn = gr.Button("Refresh", variant="secondary")
                refresh_btn.click(fn=load_non_ai_notes, inputs=[], outputs=[notes_display])
            
            # ============================================================
            # PART A: Sigmoid vs Step Activation
            # ============================================================
            with gr.Tab("Part A: Sigmoid vs Step Activation"):
                gr.Markdown("""
                ### Part A: Comparing Activation Functions (Snorlax vs Mudkip)
                
                **Goal**: Show the difference between **Sigmoid** and **Step** activation functions by training two neural networks on the same data and comparing training/validation results and predictions.
                
                **Why it matters**: Like we learned in class the step function has a hard threshold, which makes gradient-based learning difficult. The sigmoid is smooth and differentiable, so it is a better choice for training the model. Something I noticed when working on this is that with such a small dataset theres going to be some **overfitting**. Where there is high training accuracy and lower validation accuracy. You should see that the **sigmoid** model does much better on both training and validation than the **step** model, which struggles to learn. On almost all images the Step prediction was Snorlax, but the Sigmoid prediction was better at predicting.
                
                **Formulas**:
                - **Linear combination**: z = wᵀx + b  (w = weights, x = input, b = bias)
                - **Step**: y = 1 if z > 0 else 0  (This is ahard threshold)
                - **Sigmoid**: y = σ(z) = 1 / (1 + e^(-z))  (This creates a smooth curve, and values between 0 and 1)
                
                **Fixed parameters (same for both models):**
                - **2 hidden layers**, width = 64
                - Learning rate: 0.001, Epochs: 30
                - Data: Snorlax vs Mudkip (images normalized, same dimensions, grayscale)
                """)
                
                augmentation = gr.Checkbox(
                    value=True,
                    label="Enable Data Augmentation",
                    info="Random flips, rotations, and shifts (applied to both models)"
                )
                train_button = gr.Button("Train Both Models Simultaneously", variant="primary", size="lg")
                
                results_dataframe = gr.Dataframe(
                    value=pd.DataFrame(columns=["Model", "Final train acc", "Final val acc"]),
                    label="Part A: Training Results (Sigmoid vs Step)",
                    interactive=False,
                )
                
                gr.Markdown("### Training vs Validation Accuracy")
                plot_output = gr.Plot(label="Compare Sigmoid and Step")
                
                gr.Markdown("### Confusion Matrices")
                with gr.Row():
                    cm_sigmoid_output = gr.Plot(label="Sigmoid model")
                    cm_step_output = gr.Plot(label="Step model")
                
                gr.Markdown("---")
                gr.Markdown("### Try it: Upload an image to see how each model predicts (Snorlax or Mudkip)")
                
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Upload a Pokémon image")
                        predict_button = gr.Button("Predict with both models", variant="secondary")
                    with gr.Column():
                        prediction_output = gr.Textbox(label="Predictions", lines=12)
                
                train_button.click(
                    fn=train_both_models_part_a,
                    inputs=[augmentation],
                    outputs=[results_dataframe, plot_output, cm_sigmoid_output, cm_step_output]
                )
                predict_button.click(
                    fn=predict_image_both,
                    inputs=[image_input],
                    outputs=[prediction_output]
                )
            
            # ============================================================
            # PART B.0: Pixel Character
            # ============================================================
            with gr.Tab("Part B.0: Pixel Character"):
                gr.Markdown("""
                ### Part B.0: Pixel Snorlax (32×32)
                
                **Intuition**: Pixelization transforms high-resolution images into low-dimensional 
                feature vectors that neural networks can process efficiently.
                
                **Why pixelization matters for ML inputs**:
                
                - **Dimensionality reduction**: Reduces from high-resolution images (millions of pixels) 
                  to manageable feature vectors (1,024 features for 32×32)
                
                - **Computational efficiency**: 32×32 = 1,024 features vs. millions in high-res images. 
                  This makes training feasible on standard hardware.
                
                - **Standardization**: Fixed-size inputs (32×32) enable batch processing and consistent 
                  model architecture across all images.
                
                - **Feature extraction**: Forces model to learn essential patterns and shapes, 
                  not pixel-level details that may not generalize.
                
                **Mathematical framing**:
                - Input: Image I of size H×W×C (Height × Width × Channels)
                - Pixelization: Resize to 32×32, convert to grayscale → 32×32×1
                - Flattening: Reshape to 1D vector x ∈ ℝ¹⁰²⁴
                - Feature vector: x = [p₁, p₂, ..., p₁₀₂₄] where each pᵢ ∈ [0, 1] is normalized pixel intensity
                """)
                
                # Load both original and pixelated Snorlax
                snorlax_img = load_pixel_snorlax()
                original_img = load_original_snorlax()
                
                if snorlax_img is not None:
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Image Comparison")
                            
                            # Show original (unmodified) image
                            if original_img is not None:
                                original_display = gr.Image(
                                    value=original_img,
                                    label=f"Original 143.png ({original_img.size[0]}×{original_img.size[1]})",
                                    type="pil"
                                )
                            
                            if original_img is not None:
                                original_pixels = original_img.size[0] * original_img.size[1]
                                pixelated_pixels = snorlax_img.size[0] * snorlax_img.size[1]
                                reduction_factor = original_pixels / pixelated_pixels
                                
                                gr.Markdown(f"""
                                **Comparison**:
                                - **Original**: {original_img.size[0]}×{original_img.size[1]} = {original_pixels:,} pixels
                                - **Pixelated**: {snorlax_img.size[0]}×{snorlax_img.size[1]} = {pixelated_pixels} pixels
                                - **Reduction factor**: {reduction_factor:.0f}× fewer features!
                                
                                **Pixelated Image Specifications**:
                                - **Dimensions**: {snorlax_img.size[0]}×{snorlax_img.size[1]} pixels
                                - **Flattened vector dimension**: {snorlax_img.size[0] * snorlax_img.size[1]} features
                                - **Format**: RGB → converted to grayscale for ML input
                                - **Normalization**: Pixel intensities scaled to [0, 1] range
                                """)
                            else:
                                gr.Markdown(f"""
                                **Image Specifications**:
                                - **Dimensions**: {snorlax_img.size[0]}×{snorlax_img.size[1]} pixels
                                - **Total pixels**: {snorlax_img.size[0] * snorlax_img.size[1]} pixels
                                - **Flattened vector dimension**: {snorlax_img.size[0] * snorlax_img.size[1]} features
                                - **Format**: RGB → converted to grayscale for ML input
                                - **Normalization**: Pixel intensities scaled to [0, 1] range
                                
                                **Comparison**:
                                - High-res image (e.g., 1920×1080): 2,073,600 features
                                - Pixelated (32×32): 1,024 features
                                - **Reduction factor**: ~2,000× fewer features!
                                """)
                        
                        with gr.Column():
                            # Create and display visualization
                            viz_fig = visualize_pixel_character(snorlax_img)
                            visualization_plot = gr.Plot(
                                value=viz_fig,
                                label="Pixel Character Visualizations"
                            )
                else:
                    gr.Markdown("⚠️ Pixel Snorlax image not found. Please ensure the file exists at `data/snorlax_pixelated/143.png`")
            
            # ============================================================
            # PART B.1: Decision Boundary
            # ============================================================
            with gr.Tab("Part B.1: Decision Boundary"):
                
                gr.Markdown("""
                ## Part B.1: Decision Boundary Visualization
                ### Errors in Perceptron Lab Code
                
                **The Perceptron Lab notebook code has the following errors:**
                
                1. **Hardcoded Range [0, 10]**:
                   - The code uses `np.linspace(0, 10, 100)` for both x₁ and x₂
                   - This range is hardcoded and doesn't adapt to the actual data
                   - If data is outside [0, 10], the boundary won't be visible or will be incomplete
                
                2. **Boundary Line Filtering**:
                   - The code filters the boundary: `valid = (x2_line >= 0) & (x2_line <= 10)`
                   - This cuts off the decision boundary line if it extends beyond [0, 10]
                   - The boundary should extend across the entire visible plot area
                
                3. **No Data Range Adaptation**:
                   - The code doesn't check the actual data range
                   - It assumes data is always in [0, 10], which may not be true
                   - This causes misalignment between data and boundary visualization
                
                **Corrected Implementation**:
                - Uses adaptive range based on actual data: `X[:, 0].min() - 0.5` to `X[:, 0].max() + 0.5`
                - Plots the full decision boundary line without filtering
                - Ensures boundary is always visible and correctly positioned relative to data
                """)
                
                show_error_version = gr.Checkbox(
                    value=True,
                    label="Show Perceptron Lab Version (with errors) for comparison",
                    info="Compare the original code with errors vs corrected version"
                )
                
                visualize_button = gr.Button("Generate Decision Boundary Visualization", variant="primary", size="lg")
                
                boundary_plot = gr.Plot(label="Decision Boundary Visualization")
                
                gr.Markdown("---")
                
                # Connect button to function (using fixed parameters)
                def generate_visualization(show_error):
                    return visualize_decision_boundary_interactive(
                        n_samples=200, noise=0.1, epochs=100, lr=0.01, show_error=show_error
                    )
                
                visualize_button.click(
                    fn=generate_visualization,
                    inputs=[show_error_version],
                    outputs=[boundary_plot]
                )
            
            # ============================================================
            # PART B.2: Knowledge Check
            # ============================================================
            with gr.Tab("Part B.2: Knowledge Check"):
                gr.Markdown("""
                ### Part B.2: Knowledge Check Questions
                """)
                
                gr.Markdown("""
                **Questions to be answered**:
                1. What does a perceptron compute before applying the activation function?
                 - A weighted sum plus bias (coorect)
                2. What's the key difference between step and sigmoid activation?
                 - Step outputs 0 or 1 sharply; sigmoid outputs smooth values between 0-1 (correct)
                3. Why can't a single perceptron solve the XOR problem?
                 - A single perceptron can only create linear decision boundaries (Correct, Had to think about this one for a second, but a single perceptron can only create a yes or no decision boundary)
                4. What role does the bias term play?
                 - It shifts the decision boundary, allowing it to not pass through origin (Correct, I remmebr in class saying how we can set the bias initially if we have an idea of the output of the data)
                5. How does a perceptron 'learn' the correct weights?
                 - By adjusting weights to reduce prediction errors (gradient descent) (Correct, In class we discussed how reducing error/loss was the number one priority)
            
                """)
            
            # ============================================================
            # PART B.3: Star Wars Death Prediction
            # ============================================================
            with gr.Tab("Part B.3: Death Prediction (Star Wars)"):
                gr.Markdown("""
                ### Part B.3: Do Physical Attributes Predict Character Death?
                
                **Intuition**: Can we predict whether a Star Wars character will die during the movies 
                based on their physical attributes? This is a binary classification problem.
                
                **Formal ML Framing**:
                - **Inputs**: Physical attributes (height, weight, year_born, gender, species, hair_color, eye_color, skin_color, homeworld)
                - **Outputs**: Binary classification (Died = 1 if year_died is not null, Alive = 0)
                - **Labels**: Binary (0 = alive, 1 = died)
                
                **Structure and Meaning of Data** (which columns to use and why):
                - **id, name**: Not needed for prediction; identifiers only.
                - **species, gender**: Valuable (e.g. droids may be more likely to die; statistical significance).
                - **height, weight**: Yes — more body to be hit (assumption).
                - **hair_color**: Lower weight; possible camouflage effect.
                - **eye_color, skin_color**: Medium weight; may relate to survival.
                - **year_born**: High weight; older characters may be more likely to die.
                - **homeworld**: Valuable (e.g. Death Star, world-specific risks).
                - **year_died**: Used to define the target (validation: predicted died vs. non-null year_died).
                - **description**: Excluded; would require NLP.
                
                **Mathematical Formulation**:
                - **Linear Model**: z = wᵀx + b, P(died|x) = σ(z) = 1/(1 + e^(-z))
                - **Neural Network**: z₁ = ReLU(W₁x + b₁), z₂ = ReLU(W₂z₁ + b₂), P(died|x) = σ(W₃z₂ + b₃)
                - **Loss**: Binary Cross-Entropy: L = -[y log(ŷ) + (1-y) log(1-ŷ)]
                
                **Modeling Choices**:
                - **Loss**: Binary cross-entropy (BCE), as in class.
                - **Optimizer**: Adam (adaptive learning rate).
                - **Charts**: Training loss and accuracy over epochs; evaluation charts (metrics bar, ROC curves, confusion matrices).
                - **Preprocessing**: StandardScaler for numericals, LabelEncoder for categoricals. Train/test 80/20 stratified.
                
                **Model Architectures**:
                1. **Linear Regression (Logistic)**: Single layer, learns linear decision boundary.
                2. **Neural Network**: 3-layer feed-forward (64 → 32 → 1), ReLU activations, Dropout (0.2) for regularization.
                
                **Why These Choices**:
                - **Linear model**: Baseline, interpretable, fast to train.
                - **Neural network**: Can learn non-linear patterns, more expressive.
                - **Dropout**: Prevents overfitting on small dataset.
                - **Stratified split**: Maintains class distribution (important for imbalanced data).
                """)
                
                b3_state = gr.State(value=None)
                train_button = gr.Button("Train Models & Compare", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        results_text = gr.Textbox(
                            label="Results",
                            lines=25,
                            max_lines=30
                        )
                    
                    with gr.Column():
                        comparison_plot = gr.Plot(label="Model Comparison")
                
                train_button.click(
                    fn=train_and_compare_models,
                    inputs=[],
                    outputs=[results_text, comparison_plot, b3_state]
                )
                
                gr.Markdown("---")
                gr.Markdown("""
                **Create a character and predict if they die.**  
                Fill in the attributes below and click **Predict**. Train the models first if you haven’t already.
                """)
                
                cat_options = get_star_wars_categorical_options()
                with gr.Row():
                    with gr.Column():
                        pred_height = gr.Number(label="Height (m)", value=1.7)
                        pred_weight = gr.Number(label="Weight (kg)", value=70)
                        pred_year_born = gr.Number(label="Year born (BBY)", value=50)
                        pred_gender = gr.Dropdown(
                            label="Gender",
                            choices=cat_options['gender'],
                            value="Male" if "Male" in cat_options['gender'] else cat_options['gender'][0]
                        )
                        pred_species = gr.Dropdown(
                            label="Species",
                            choices=cat_options['species'],
                            value="Human" if "Human" in cat_options['species'] else cat_options['species'][0]
                        )
                    with gr.Column():
                        pred_hair_color = gr.Dropdown(
                            label="Hair color",
                            choices=cat_options['hair_color'],
                            value="Brown" if "Brown" in cat_options['hair_color'] else cat_options['hair_color'][0]
                        )
                        pred_eye_color = gr.Dropdown(
                            label="Eye color",
                            choices=cat_options['eye_color'],
                            value="Brown" if "Brown" in cat_options['eye_color'] else cat_options['eye_color'][0]
                        )
                        pred_skin_color = gr.Dropdown(
                            label="Skin color",
                            choices=cat_options['skin_color'],
                            value=cat_options['skin_color'][0]
                        )
                        pred_homeworld = gr.Dropdown(
                            label="Homeworld",
                            choices=cat_options['homeworld'],
                            value="Tatooine" if "Tatooine" in cat_options['homeworld'] else cat_options['homeworld'][0]
                        )
                
                predict_btn = gr.Button("Predict", variant="secondary")
                prediction_out = gr.Markdown()
                
                predict_btn.click(
                    fn=predict_character,
                    inputs=[b3_state, pred_height, pred_weight, pred_year_born, pred_gender, pred_species,
                            pred_hair_color, pred_eye_color, pred_skin_color, pred_homeworld],
                    outputs=prediction_out
                )
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Check if data directories exist
    if not SNORLAX_DIR.exists() or not MUDKIP_DIR.exists():
        print(f"Error: Data directories not found!")
        print(f"Please ensure {SNORLAX_DIR} and {MUDKIP_DIR} exist with images.")
    else:
        # Count images
        snorlax_count = len([f for f in os.listdir(SNORLAX_DIR) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        mudkip_count = len([f for f in os.listdir(MUDKIP_DIR) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        
        print(f"Found {snorlax_count} Snorlax images and {mudkip_count} Mudkip images")
        
        # Create and launch the Gradio interface
        demo = create_gradio_interface()
        demo.launch()
