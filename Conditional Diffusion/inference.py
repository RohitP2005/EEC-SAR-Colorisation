import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

from models.unet_conditional import ConditionalUNet
from models.noise_scheduler import NoiseScheduler

# ---- Config ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "sar_rgb_diffusion"
model_path = f"output/{model_name}/epoch_100.pt"
sar_image_path = "/opt/Hackathon/EEC-SAR-Colorisation/dataset/agri/s1/ROIs1868_summer_s1_59_p2.png"
image_size = 256
timesteps = 1000

# ---- Preprocessing ----
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# Load SAR image
sar_image = Image.open(sar_image_path)
sar_tensor = transform(sar_image).unsqueeze(0).to(device)  # Shape: [1, 1, H, W]

# ---- Load model and scheduler ----
model = ConditionalUNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

scheduler = NoiseScheduler(timesteps=timesteps)
betas = scheduler.betas
alphas = scheduler.alphas
alpha_bar = scheduler.alpha_cumprod

# ---- Start from random noise ----
x_t = torch.randn((1, 3, image_size, image_size), device=device)

# ---- Reverse diffusion process ----
with torch.no_grad():
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

        predicted_noise = model(x_t, sar_tensor)

        a_t = alphas[t].to(device)
        ab_t = alpha_bar[t].to(device)
        b_t = betas[t].to(device)

        x_t = (1 / torch.sqrt(a_t)) * (
            x_t - ((1 - a_t) / torch.sqrt(1 - ab_t)) * predicted_noise
        )

        if t > 0:
            noise = torch.randn_like(x_t)
            x_t += torch.sqrt(b_t) * noise

# ---- Save or show output image ----
output = x_t.squeeze(0).clamp(0, 1).cpu()
output_img = transforms.ToPILImage()(output)

# Save and Show
output_path = f"{model_name}_generated_rgb.png"
output_img.save(output_path)
output_img.show()
print(f"[✓] Generated RGB image saved to: {output_path}")
