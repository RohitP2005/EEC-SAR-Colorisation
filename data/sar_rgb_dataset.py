import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class SARToRGBDataset(Dataset):
    def __init__(self, sar_dir, rgb_dir, image_size=256):
        self.sar_dir = sar_dir
        self.rgb_dir = rgb_dir

        # Get list of SAR files (assumes filenames contain '_s1_')
        self.sar_files = sorted([
            f for f in os.listdir(sar_dir)
            if '_s1_' in f and f.endswith('.png')
        ])

        # Define transformations for SAR (grayscale) and RGB images
        self.transform_sar = T.Compose([
            T.Resize((image_size, image_size)),
            T.Grayscale(),  # Ensure single channel
            T.ToTensor(),
        ])

        self.transform_rgb = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # Will be 3-channel automatically
        ])

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        # Get the SAR filename and derive the corresponding RGB filename
        sar_filename = self.sar_files[idx]
        rgb_filename = sar_filename.replace('_s1_', '_s2_')

        sar_path = os.path.join(self.sar_dir, sar_filename)
        rgb_path = os.path.join(self.rgb_dir, rgb_filename)

        # Load images
        sar_image = Image.open(sar_path).convert('L')   # Load as grayscale
        rgb_image = Image.open(rgb_path).convert('RGB') # Load as RGB

        # Apply transformations
        sar_tensor = self.transform_sar(sar_image)
        rgb_tensor = self.transform_rgb(rgb_image)

        return sar_tensor, rgb_tensor
