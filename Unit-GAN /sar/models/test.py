# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# from skimage.color import lab2rgb
# import numpy as np

# # Load the trained model
# model_path = "/opt/Hackathon/SAR-RGB-Colourization/SAR/sar/models/outputs/checkpoints/model_epoch_20.pth"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # FIXED: Load model correctly by using torch.load with strict=False
# state_dict = torch.load(model_path, map_location=device)
# from model import UnetGenerator  # FIXED: Ensure the correct model class is imported

# model = UnetGenerator()
# model.load_state_dict(state_dict, strict=False)  # FIXED: Allow missing keys if necessary
# model = model.to(device)
# model.eval()

# def preprocess_image(image_path):
#     img = Image.open(image_path).convert("L")  # Load as grayscale
#     transform = transforms.Compose([
#         transforms.Resize((256, 256), Image.BICUBIC),  # FIXED: Use BICUBIC interpolation
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))  # FIXED: Ensure normalization matches training
#     ])
#     img = transform(img).unsqueeze(0)  # Add batch dimension -> (1,1,256,256)
#     return img.to(device)

# def postprocess_output(l_channel, ab_channels):
#     l_channel = (l_channel + 1.0) / 2.0  # Convert L back to [0,1] range
#     ab_channels = ab_channels * 110  # FIXED: Reverse ab normalization
#     lab = torch.cat([l_channel, ab_channels], dim=1).squeeze(0).permute(1, 2, 0).cpu().numpy()
#     rgb = lab2rgb(lab)  # Convert to RGB
#     return (rgb * 255).astype(np.uint8)  # Convert to 8-bit image

# def colorize_image(image_path, output_path):
#     L = preprocess_image(image_path)  # Preprocess input
#     with torch.no_grad():
#         fake_ab = model(L)  # Get predicted ab channels
    
#     rgb_img = postprocess_output(L, fake_ab)  # Convert model output to RGB
#     Image.fromarray(rgb_img).save(output_path)  # Save the output

# # Example usage
# test_image_path = "/opt/Hackathon/SAR-RGB-Colourization/SAR/v_2/agri/s1/ROIs1868_summer_s1_59_p2.png"
# output_image_path = "/opt/Hackathon/SAR-RGB-Colourization/SAR/sar/models/outputs/generated_images/generated_img.png"
# colorize_image(test_image_path, output_image_path)
# print(f"Colorized image saved to {output_image_path}")

















# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# from skimage.color import lab2rgb
# import numpy as np
# from model import UnetGenerator

# # Paths
# model_path = "/opt/Hackathon/SAR-RGB-Colourization/SAR/sar/models/outputs/checkpoints/best_model.pth"
# test_image_path = "/opt/Hackathon/SAR-RGB-Colourization/SAR/v_2/agri/s1/ROIs1868_summer_s1_59_p2.png"
# output_image_path = "/opt/Hackathon/SAR-RGB-Colourization/SAR/sar/models/outputs/generated_images/generated_img.png"

# # Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load trained model
# print("Loading model...")
# try:
#     model = UnetGenerator(input_c=1, output_c=2)  # Ensure correct parameter names
#     checkpoint = torch.load(model_path, map_location=device)
    
#     if 'net_G' in checkpoint:
#         checkpoint = checkpoint['net_G']  # Extract generator weights if wrapped in dict

#     model.load_state_dict(checkpoint)
#     model.to(device)
#     model.eval()
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)

# # Image preprocessing
# def preprocess_image(image_path):
#     img = Image.open(image_path).convert("L")  # Convert to grayscale
#     original_size = img.size  # Store original size for later resizing
    
#     transform = transforms.Compose([
#         transforms.Resize((256, 256), Image.BICUBIC),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
    
#     img = transform(img).unsqueeze(0)  # Add batch dimension -> (1,1,256,256)
#     return img.to(device), original_size

# # Convert model output to RGB
# def postprocess_output(l_channel, ab_channels, original_size):
#     l_channel = (l_channel + 1.0) / 2.0  # Convert L back to [0,1] range
#     ab_channels = ab_channels * 110  # Reverse ab normalization

#     lab = torch.cat([l_channel, ab_channels], dim=1).squeeze(0).permute(1, 2, 0).cpu().numpy()
#     rgb = lab2rgb(lab)  # Convert to RGB
    
#     rgb_img = (rgb * 255).astype(np.uint8)  # Convert to 8-bit image
#     rgb_img = Image.fromarray(rgb_img).resize(original_size, Image.BICUBIC)  # Resize back to original size
    
#     return rgb_img

# # Colorization function
# def colorize_image(image_path, output_path):
#     try:
#         L, original_size = preprocess_image(image_path)  # Preprocess input
#         with torch.no_grad():
#             fake_ab = model(L)  # Get predicted ab channels
        
#         rgb_img = postprocess_output(L, fake_ab, original_size)  # Convert to RGB
#         rgb_img.save(output_path)  # Save the output
        
#         print(f"Colorized image saved to {output_path}")
#     except Exception as e:
#         print(f"Error in processing: {e}")

# # Run colorization
# colorize_image(test_image_path, output_image_path)












# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# from skimage.color import lab2rgb
# import numpy as np
# from model import UnetGenerator

# # Paths
# model_path = "/opt/Hackathon/SAR-RGB-Colourization/SAR/sar/models/outputs/checkpoints/model_epoch_20.pth"
# test_image_path = "/opt/Hackathon/SAR-RGB-Colourization/SAR/v_2/agri/s1/ROIs1868_summer_s1_59_p2.png"
# output_image_path = "/opt/Hackathon/SAR-RGB-Colourization/SAR/sar/models/outputs/generated_images/generated_img.png"

# # Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load trained model
# model_instance = UnetGenerator()
# model_state = torch.load(model_path, map_location=device)
# model_instance.load_state_dict(model_state, strict=False)
# model = model_instance.to(device)
# model.eval()

# # Image preprocessing
# def preprocess_image(image_path):
#     img = Image.open(image_path).convert("L")  # Convert to grayscale
#     original_size = img.size  # Store original size for later resizing
    
#     transform = transforms.Compose([
#         transforms.Resize((256, 256), Image.BICUBIC),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
    
#     img = transform(img).unsqueeze(0)  # Add batch dimension -> (1,1,256,256)
#     return img.to(device), original_size

# # Convert model output to RGB
# def postprocess_output(l_channel, ab_channels, original_size):
#     l_channel = (l_channel + 1.0) / 2.0  # Convert L back to [0,1] range
#     ab_channels = ab_channels * 110  # Reverse ab normalization

#     lab = torch.cat([l_channel, ab_channels], dim=1).squeeze(0).permute(1, 2, 0).cpu().numpy()
#     rgb = lab2rgb(lab)  # Convert to RGB
    
#     rgb_img = (rgb * 255).astype(np.uint8)  # Convert to 8-bit image
#     rgb_img = Image.fromarray(rgb_img).resize(original_size, Image.BICUBIC)  # Resize back to original size
    
#     return rgb_img

# # Colorization function
# def colorize_image(image_path, output_path):
#     try:
#         L, original_size = preprocess_image(image_path)  # Preprocess input
#         with torch.no_grad():
#             fake_ab = model(L)  # Get predicted ab channels
        
#         rgb_img = postprocess_output(L, fake_ab, original_size)  # Convert to RGB
#         rgb_img.save(output_path)  # Save the output
        
#         print(f"Colorized image saved to {output_path}")
#     except Exception as e:
#         print(f"Error in processing: {e}")

# # Run colorization
# colorize_image(test_image_path, output_image_path)






import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage.color import lab2rgb
from model import MainModel  # Import your trained model class

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model
model = MainModel()
checkpoint = torch.load("/opt/Hackathon/SAR-RGB-Colourization/SAR/sar/models/outputs/checkpoints/model_epoch_20.pth")
print(checkpoint.keys())  # Check if 'model_state_dict' is present
model.load_state_dict(torch.load("/opt/Hackathon/SAR-RGB-Colourization/SAR/sar/models/outputs/checkpoints/model_epoch_20.pth", map_location=device))
model = model.to(device)
model.eval()

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor()
])

def preprocess_image(image_path):
    """Load and preprocess a single SAR image."""
    img = Image.open(image_path).convert("L")  # Convert to grayscale (SAR)
    img = transform(img)  # Resize and convert to tensor
    img = img * 2.0 - 1.0  # Normalize L channel to [-1, 1]
    return img.unsqueeze(0).to(device)  # Add batch dimension

def postprocess_output(L, ab):
    """Convert L and ab channels back to an RGB image."""
    L = (L + 1.0) * 50.0  # Denormalize L channel
    ab = ab * 110.0  # Denormalize ab channels
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().detach().numpy()
    
    rgb_images = []
    for img in Lab:
        rgb_images.append(lab2rgb(img.astype('float64')))
    return np.stack(rgb_images, axis=0)

def colorize_image(image_path, output_path):
    """Run inference on a single SAR image and save the result."""
    # Preprocess input image
    L = preprocess_image(image_path)

    # Generate colorized image
    with torch.no_grad():
        ab_fake = model.net_G(L)  # Generate ab channels
    
    # Convert to RGB
    rgb_image = postprocess_output(L, ab_fake)[0]

    # Save the output
    output_img = Image.fromarray((rgb_image * 255).astype(np.uint8))
    output_img.save(output_path)
    print(f"Colorized image saved to {output_path}")

    # Display the results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(Image.open(image_path).convert("L"), cmap="gray")
    ax[0].set_title("Input SAR Image")
    ax[0].axis("off")
    
    ax[1].imshow(rgb_image)
    ax[1].set_title("Colorized RGB Image")
    ax[1].axis("off")
    
    plt.show()

# Test the model on a new SAR image
test_image_path = "/opt/Hackathon/SAR-RGB-Colourization/SAR/v_2/agri/s1/ROIs1868_summer_s1_59_p2.png"  # Replace with your SAR image path
output_image_path = "/opt/Hackathon/SAR-RGB-Colourization/SAR/sar/models/outputs/generated_images/colorized_output_2.png"
colorize_image(test_image_path, output_image_path)





