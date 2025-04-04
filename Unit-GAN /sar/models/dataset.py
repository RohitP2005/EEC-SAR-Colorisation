# Cell 1: Imports and Setup
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from skimage import io
import torchvision
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm

import torch
from torch import nn, optim

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the base directory of your dataset
base_dir = "/opt/Hackathon/SAR-RGB-Colourization/SAR/v_2"
categories = ['agri', 'barrenland', 'grassland', 'urban']

# Collect all pairs of input (SAR) and output (optical) image paths
input_output_pairs = []
for category in categories:
    input_folder = os.path.join(base_dir, category, 's1')
    output_folder = os.path.join(base_dir, category, 's2')
    
    # Get all input and output image file paths
    input_images = sorted(glob.glob(os.path.join(input_folder, "*.png")))
    output_images = sorted(glob.glob(os.path.join(output_folder, "*.png")))
    
    # Ensure that the number of images match
    assert len(input_images) == len(output_images), \
        f"Number of images in {input_folder} and {output_folder} do not match."
    
    for input_img, output_img in zip(input_images, output_images):
        input_output_pairs.append((input_img, output_img))

# Checking the size of the dataset
print(f"Total dataset size: {len(input_output_pairs)}")

# Shuffle and split the dataset into training and validation sets
np.random.seed(123)  # Seeding for reproducibility
input_output_pairs = np.random.permutation(input_output_pairs)  # Shuffling the pairs

# Splitting into training and validation sets
train_ratio = 0.8
num_total = len(input_output_pairs)
num_train = int(train_ratio * num_total)

train_pairs = input_output_pairs[:num_train]
val_pairs = input_output_pairs[num_train:]

print(f"Training set size: {len(train_pairs)}")
print(f"Validation set size: {len(val_pairs)}")

# Example: Accessing a pair
example_input, example_output = train_pairs[0]
print("Example input image path:", example_input)
print("Example output image path:", example_output)


class ColorizationDataset(Dataset):
    def __init__(self, pairs, split='train', transform=None):
        self.pairs = pairs
        self.split = split
        self.size = 256  # Image size
        self.transform = transform

        # Define default transforms if none are provided
        if self.transform is None:
            if self.split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((self.size, self.size), Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                ])
            else:
                self.transform = transforms.Resize((self.size, self.size), Image.BICUBIC)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_path, output_path = self.pairs[idx]
        
        # Load the input (SAR) image and convert to grayscale
        input_img = Image.open(input_path).convert('L')  # Ensure it's grayscale
        input_img = self.transform(input_img)
        input_img = transforms.ToTensor()(input_img)  # Shape: [1, H, W]
        
        # Load the output (optical) image and convert to RGB
        output_img = Image.open(output_path).convert('RGB')
        output_img = self.transform(output_img)
        output_img = transforms.ToTensor()(output_img)  # Shape: [3, H, W]
        
        # Convert output image from RGB to Lab color space
        output_img_np = output_img.permute(1, 2, 0).numpy()  # Convert to HWC
        output_lab = rgb2lab(output_img_np).astype('float32')
        output_lab = transforms.ToTensor()(output_lab)  # Shape: [3, H, W]
        
        # Normalize L and ab channels
        L = input_img * 2.0 - 1.0  # Normalize L channel to [-1, 1]
        ab = output_lab[1:, ...] / 128.  # Normalize ab channels to [-1, 1]
        
        return {'L': L, 'ab': ab}

def make_dataloaders(pairs, batch_size=8, num_workers=4, split='train'):
    dataset = ColorizationDataset(pairs, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split=='train'),
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

# Create data loaders
batch_size = 8  # Reduced batch size to fit in memory
train_dl = make_dataloaders(train_pairs, batch_size=batch_size, num_workers=4, split='train')
val_dl = make_dataloaders(val_pairs, batch_size=batch_size, num_workers=4, split='val')