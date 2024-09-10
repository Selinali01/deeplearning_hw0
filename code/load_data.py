import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import TensorDataset, random_split
from utils.mnist_reader import load_mnist
import numpy as np

def load_mnist_data():
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')

    # Load training data
    train_images, train_labels = load_mnist(data_dir, kind='train')

    # Load test data
    test_images, test_labels = load_mnist(data_dir, kind='test')

    # Make writable copies of the arrays
    train_images = np.array(train_images, copy=True)
    train_labels = np.array(train_labels, copy=True)
    test_images = np.array(test_images, copy=True)
    test_labels = np.array(test_labels, copy=True)

    # Convert to PyTorch tensors and reshape images
    train_images = torch.from_numpy(train_images).float().view(-1, 1, 28, 28) / 255.0
    train_labels = torch.from_numpy(train_labels).long()
    test_images = torch.from_numpy(test_images).float().view(-1, 1, 28, 28) / 255.0
    test_labels = torch.from_numpy(test_labels).long()

    # Normalize the data
    mean = train_images.mean()
    std = train_images.std()
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

    # Create datasets
    full_train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    # Split the training data into train and validation sets
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset

'''
if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = load_mnist_data()
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {len(set(train_dataset.dataset.tensors[1].tolist()))}")
    print(f"Shape of a single image: {train_dataset.dataset.tensors[0][0].shape}")

    '''