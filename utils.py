# Import libries
import os
import torch
from torchvision import datasets, transforms
import json
from PIL import Image
import numpy as np

import pickle

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def get_dataloaders(data_dir, batch_size):
    # Directories for training, validation, and testing datasets
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Transforms for the training, validation, and testing sets
    data_transforms_train = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms_test = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms_valid = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    image_datasets_train = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    image_datasets_test = datasets.ImageFolder(test_dir, transform=data_transforms_test)
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)

    # Using the datasets and transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(image_datasets_test, batch_size=batch_size, shuffle=False)
    validloader = torch.utils.data.DataLoader(image_datasets_valid, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, validloader, image_datasets_train.class_to_idx


#  load in a mapping from category label to category name
def load_category_names(filepath):
    ''' Loads a JSON file containing category-to-name mappings.'''
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def save_checkpoint(model, optimizer, epoch, class_to_idx, checkpoint_path):
    """
    Saves the model checkpoint including:
    - Model state dict
    - Optimizer state dict
    - Epoch
    - Class-to-idx mapping
    """
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved as {checkpoint_path}")


def load_checkpoint(filepath, model, optimizer):
    """
    Loads the model checkpoint from the specified filepath.

    Args:
        filepath (str): Path to the checkpoint file.
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.

    Returns:
        model: The model with the loaded state.
        optimizer: The optimizer with the loaded state.
        epoch: The epoch the training stopped at.
        class_to_idx: Mapping of classes to indices.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch, model.class_to_idx


def process_image(image_path):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model,
       returns a Numpy array.
    '''

    # Open the image
    img = Image.open(image_path)

    # Resize the image
    # Resize where the shortest side is 256 pixels, maintaining the aspect ratio
    if img.size[0] > img.size[1]:
        img.thumbnail((256, 256 * img.size[0] // img.size[1]))
    else:
        img.thumbnail((256 * img.size[1] // img.size[0], 256))

    # Crop the center of the image to 224x224
    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))

    # Convert the image to a numpy array and normalize
    np_image = np.array(img) / 255.0

    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions so that color channel is first since PyTorch expects it
    np_image = np_image.transpose((2, 0, 1))

    return np_image

# Function to save metrics to a file
import pickle
import os

# Function to save metrics and confusion matrix to a file
def save_metrics_to_file(filename, epoch_n, train_losses, valid_losses, train_acc, confusion_matrix=None):
    # Check if the file already exists
    if os.path.exists(filename):
        # Load existing data
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            # Append new metrics to existing lists
            data['epoch_n'].extend(epoch_n)
            data['train_losses'].extend(train_losses)
            data['valid_losses'].extend(valid_losses)
            data['train_acc'].extend(train_acc)

    else:
        # If the file doesn't exist, create a new dictionary
        data = {
            'epoch_n': epoch_n,
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'train_acc': train_acc,
              }

    # Save the combined data to the file
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
