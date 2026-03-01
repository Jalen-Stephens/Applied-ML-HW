# Loader
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class MuseumDataset(Dataset):
    """Custom dataset: images + CSV metadata."""
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.metadata = pd.read_csv(csv_path)
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        # return the number of images
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # apply self.transform if it exists
        if self.transform is not None:
            image = self.transform(image)

        # Return image + row index (look up metadata later)
        return image, idx

# TODO: define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    ),
])

# ---- fill these in with your actual paths ----
train_image_dir = "HW2/data_oceania_HW2/train"        # e.g., "data/train"
train_csv_path  = "HW2/data_oceania_HW2/metadata.csv" # e.g., "data/metadata.csv"

train_dataset = MuseumDataset(train_image_dir, train_csv_path, transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# Visualize a batch
import matplotlib.pyplot as plt

def show_batch(loader, n=8):
    images, idxs = next(iter(loader))
    fig, axes = plt.subplots(1, min(n, len(images)), figsize=(2*n, 2))
    for i in range(min(n, len(images))):
        img = images[i].permute(1, 2, 0)
        # Undo normalization for display
        img = img * torch.tensor([0.229, 0.224, 0.225]) + \
                     torch.tensor([0.485, 0.456, 0.406])
        img = img.clamp(0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

show_batch(train_loader)


#CNN

import numpy as np
import torch
import torch.nn as nn

num_layers = np.random.randint(2, 6)
channels = [3] + [np.random.choice([16, 32, 64]) for _ in range(num_layers)]

layers = []
for i in range(num_layers):
    layers += [
        nn.Conv2d(
            in_channels=channels[i],
            out_channels=channels[i+1],
            kernel_size=3,
            padding=1
        ),
        nn.BatchNorm2d(channels[i+1]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)  # spatial downsampling
    ]

layers.append(nn.AdaptiveAvgPool2d(1))  # global average pooling
layers.append(nn.Flatten())

random_cnn = nn.Sequential(*layers)

# Test it
x = torch.randn(1, 3, 224, 224)
embedding = random_cnn(x)
print(f"Layers: {num_layers}, Embedding dim: {embedding.shape[1]}")


# embedding extraction

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---- Extract embeddings (no gradients needed!) ----
embeddings_list = []
indices_list = []

random_cnn.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_cnn = random_cnn.to(device)

with torch.no_grad():
    for images, idxs in train_loader:
        images = images.to(device)
        emb = random_cnn(images)          # pass images through the CNN
        embeddings_list.append(emb.cpu()) # move to cpu for numpy later
        indices_list.append(idxs)

embeddings = torch.cat(embeddings_list, dim=0).numpy()
all_idxs    = torch.cat(indices_list, dim=0).numpy()

# ---- Use metadata for coloring ----
metadata = train_dataset.metadata

# Change this to: 'artist_culture', 'region', or 'medium_materials'
color_col = "artist_culture"

label_strings = metadata[color_col].iloc[all_idxs].astype(str).values
unique_labels = list(dict.fromkeys(label_strings))  # stable unique labels
color_ids = np.array([unique_labels.index(l) for l in label_strings])

# ---- Run all three methods ----
pca_2d  = PCA(n_components=2).fit_transform(embeddings)
tsne_2d = TSNE(n_components=2, perplexity=30, init="random", learning_rate="auto").fit_transform(embeddings)
umap_2d = umap.UMAP(n_components=2).fit_transform(embeddings)

# ---- Plot side by side ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, data, title in zip(
    axes,
    [pca_2d, tsne_2d, umap_2d],
    ["PCA", "t-SNE", "UMAP"]
):
    scatter = ax.scatter(data[:, 0], data[:, 1], c=color_ids, cmap="tab10", s=8, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

plt.colorbar(scatter, ax=axes[-1])
plt.suptitle(f"Colored by: {color_col}")
plt.tight_layout()
plt.show()