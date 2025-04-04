import os
import torch
from torch.utils.data import DataLoader
from models.unet_conditional import ConditionalUNet
from models.noise_scheduler import NoiseScheduler
from data.sar_rgb_dataset import SARToRGBDataset
import tqdm

# ---- Config ----
MODEL_NAME = "sar_rgb_diffusion"   # 🔁 Change this to whatever you want
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
IMAGE_SIZE = 256

# ---- Device Setup ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Create output directory ----
output_dir = os.path.join("output", MODEL_NAME)
os.makedirs(output_dir, exist_ok=True)

# ---- Dataset & DataLoader ----
dataset = SARToRGBDataset(
    sar_dir="dataset/agri/s1",
    rgb_dir="dataset/agri/s2",
    image_size=IMAGE_SIZE
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- Model, Optimizer, Scheduler ----
model = ConditionalUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = NoiseScheduler()

# ---- Training Loop ----
for epoch in range(EPOCHS):
    epoch_loss = 0
    for sar, rgb in tqdm.tqdm(loader, desc=f"Epoch {epoch+1}"):
        sar = sar.to(device)
        rgb = rgb.to(device)

        # Add noise to RGB at random timestep
        t = torch.randint(0, scheduler.timesteps, (rgb.size(0),), device=device).long()
        x_t, noise = scheduler.get_noisy_image(rgb, t)

        pred = model(x_t, sar)

        loss = torch.nn.functional.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    # Save model for this epoch
    torch.save(model.state_dict(), os.path.join(output_dir, f"epoch_{epoch+1}.pt"))

    # Also save the most recent model as best.pt
    torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))
