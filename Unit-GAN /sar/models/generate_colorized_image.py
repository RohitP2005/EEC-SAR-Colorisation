# import torch
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from utils import lab_to_rgb
# from dataset import val_dl
# from model import MainModel

# # Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the trained model
# model = MainModel()
# model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth", map_location=device))
# model.to(device)
# model.eval()

# # Convert DataLoader to a list for random access
# val_data = list(val_dl)

# # Randomly select a batch
# random_batch = random.choice(val_data)

# # Extract L and ab from the random batch
# L_batch = random_batch["L"].to(device)
# ab_real_batch = random_batch["ab"].to(device)

# # Pick a random index within the batch
# random_idx = random.randint(0, L_batch.shape[0] - 1)

# # Select the random image
# L = L_batch[random_idx:random_idx+1]  # Keep batch dimension
# ab_real = ab_real_batch[random_idx:random_idx+1]  # Keep batch dimension

# # Forward pass: Predict ab channels
# with torch.no_grad():
#     ab_fake = model.net_G(L)  # Generate color channels

# # Convert to numpy for visualization
# L = L.cpu().numpy()[0, 0, :, :]  # Remove batch & channel dimension
# ab_fake = ab_fake.cpu().numpy()[0]  # Remove batch dimension
# ab_real = ab_real.cpu().numpy()[0]  # Remove batch dimension

# # Convert L + ab to RGB
# rgb_fake = lab_to_rgb(torch.tensor(L).unsqueeze(0).unsqueeze(0), torch.tensor(ab_fake).unsqueeze(0))
# rgb_real = lab_to_rgb(torch.tensor(L).unsqueeze(0).unsqueeze(0), torch.tensor(ab_real).unsqueeze(0))

# # Plot results
# fig, axes = plt.subplots(1, 3, figsize=(12, 5))
# axes[0].imshow(L, cmap="gray")
# axes[0].set_title("Input (L channel)")
# axes[0].axis("off")

# axes[1].imshow(rgb_fake[0])  # Show colorized image
# axes[1].set_title("Predicted (Colorized)")
# axes[1].axis("off")

# axes[2].imshow(rgb_real[0])  # Show ground truth
# axes[2].set_title("Ground Truth")
# axes[2].axis("off")

# plt.show()



# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.color import rgb2lab, lab2rgb
# from skimage import io
# from torchvision import transforms
# from model import MainModel
# from utils import lab_to_rgb

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load trained model
# model = MainModel()
# model.load_state_dict(torch.load("outputs/checkpoints/model_epoch_74.pth", map_location=device))
# model.to(device)
# model.eval()

# # Define the static image path (CHANGE THIS TO YOUR IMAGE)
# IMAGE_PATH = "/opt/Hackathon/SAR-RGB-Colourization/SAR/v_2/agri/s1/ROIs1868_summer_s1_59_p2.png"  # Update this

# # Load and preprocess the image
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize to match model input size
#     transforms.ToTensor()
# ])

# image = io.imread(IMAGE_PATH)
# if image.ndim == 2:  # Convert grayscale to 3-channel grayscale
#     image = np.stack([image] * 3, axis=-1)

# image_lab = rgb2lab(image).astype("float32")  # Convert RGB to Lab
# L = image_lab[:, :, 0]  # Extract L channel
# L = (L / 50.0) - 1.0  # Normalize to [-1, 1]
# L_tensor = torch.tensor(L).unsqueeze(0).unsqueeze(0).to(device)  # Add batch & channel dim

# # Predict the ab channels
# with torch.no_grad():
#     ab_pred = model.net_G(L_tensor).cpu()  # Get predicted ab channels

# # Convert tensors to numpy for visualization
# L = L_tensor.cpu().numpy()[0, 0, :, :]
# ab_pred = ab_pred.numpy()[0]

# # Convert Lab to RGB
# rgb_pred = lab_to_rgb(torch.tensor(L).unsqueeze(0).unsqueeze(0), torch.tensor(ab_pred).unsqueeze(0))

# # Display results
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(L, cmap="gray")
# axes[0].set_title("Input (L Channel)")
# axes[0].axis("off")

# axes[1].imshow(rgb_pred[0])  # Show the colorized image
# axes[1].set_title("Predicted (Colorized)")
# axes[1].axis("off")

# plt.show()



import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2lab
from torchvision import transforms
from model import MainModel
from utils import lab_to_rgb

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = MainModel()
model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Path to SAR input image
IMAGE_PATH = "/home/projects/Desktop/test_7.jpg"  # <-- Change this to your SAR image path

# Load image
image = io.imread(IMAGE_PATH)

# Ensure it's 3-channel grayscale (for rgb2lab)
if image.ndim == 2:
    image = np.stack([image] * 3, axis=-1)

# Convert to Lab and extract L channel
image_lab = rgb2lab(image).astype("float32")
L_np = image_lab[:, :, 0]  # Shape: [H, W]
L_norm = (L_np / 50.0) - 1.0  # Normalize to [-1, 1]

# Convert to tensor
L_tensor = torch.tensor(L_norm).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, H, W]

# Predict ab channels
with torch.no_grad():
    ab_pred = model.net_G(L_tensor).cpu()

# Convert to RGB
rgb_pred = lab_to_rgb(L_tensor.cpu(), ab_pred)

# Plot SAR and predicted output
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(L_np, cmap='gray')
axes[0].set_title("Input SAR (L Channel)")
axes[0].axis("off")

axes[1].imshow(rgb_pred[0])
axes[1].set_title("Predicted Colorized Output")
axes[1].axis("off")

plt.tight_layout()
plt.show()


class SingleImageDataset(Dataset):
    def __init__(self, input_path, output_path, transform=None):
        self.input_path = input_path
        self.output_path = output_path
        self.size = 256  # Match training size
        self.transform = transform

        # Default transform (same as validation)
        if self.transform is None:
            self.transform = transforms.Resize((self.size, self.size), Image.BICUBIC)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Load SAR input
        input_img = Image.open(self.input_path).convert('L')
        input_img = self.transform(input_img)
        input_img = transforms.ToTensor()(input_img)

        # Load RGB output (ground truth)
        output_img = Image.open(self.output_path).convert('RGB')
        output_img = self.transform(output_img)
        output_img = transforms.ToTensor()(output_img)

        # Convert RGB to Lab
        output_img_np = output_img.permute(1, 2, 0).numpy()
        output_lab = rgb2lab(output_img_np).astype('float32')
        output_lab = transforms.ToTensor()(output_lab)

        # Normalize
        L = input_img * 2.0 - 1.0  # [-1, 1]
        ab = output_lab[1:, ...] / 128.  # ab channels to [-1, 1]

        return {'L': L, 'ab': ab}
