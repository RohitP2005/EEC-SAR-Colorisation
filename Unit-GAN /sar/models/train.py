# import os
# import glob
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from skimage.color import rgb2lab, lab2rgb
# from skimage import io
# import torchvision
# from torchvision import transforms
# from torchvision.models import vgg16, VGG16_Weights
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils import spectral_norm
# from utils import lab_to_rgb
# from classes import UnetBlock, UnetGenerator, PatchDiscriminatorSN, GANLoss
# from utils import AverageMeter, visualize
# import torch
# from model import MainModel
# from torch import nn, optim
# from dataset import train_dl, val_dl

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# os.makedirs("outputs/checkpoints", exist_ok=True)
# os.makedirs("outputs/images", exist_ok=True)


# def pretrain_generator(net_G, train_dl, criterion, optimizer, epochs):
#     net_G.train()
#     for epoch in range(epochs):
#         loss_meter = AverageMeter()
#         for data in tqdm(train_dl):
#             L = data['L'].to(device)
#             ab = data['ab'].to(device)

#             preds = net_G(L)
#             loss = criterion(preds, ab)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             loss_meter.update(loss.item(), L.size(0))

#         print(f"Pretraining Epoch [{epoch+1}/{epochs}], Loss: {loss_meter.avg:.5f}")


# def train_model(model, train_dl, val_dl, epochs, pretrain_epochs=5, display_every=5):
#     # Pretrain Generator
#     print("Starting Generator Pretraining...")
#     pretrain_generator(
#         net_G=model.module.net_G,
#         train_dl=train_dl,
#         criterion=model.module.L1criterion,
#         optimizer=model.module.opt_G,
#         epochs=pretrain_epochs
#     )
#     print("Pretraining Completed.\n")

#     # Training with GAN
#     for epoch in range(epochs):
#         model.module.net_G.train()
#         model.module.net_D.train()
#         loss_meter_dict = {'loss_D': AverageMeter(), 'loss_G': AverageMeter()}
#         for data in tqdm(train_dl):
#             model.module.setup_input(data)
#             model.module.optimize()

#             # Update loss meters
#             loss_meter_dict['loss_D'].update(model.module.loss_D.item(), data['L'].size(0))
#             loss_meter_dict['loss_G'].update(model.module.loss_G.item(), data['L'].size(0))

#         # Validation and Visualization
#         if (epoch + 1) % display_every == 0:
#             print(f"\nEpoch [{epoch+1}/{epochs}]")
#             print(f"Loss_D: {loss_meter_dict['loss_D'].avg:.5f}, "
#                   f"Loss_G: {loss_meter_dict['loss_G'].avg:.5f}")
#             data = next(iter(val_dl))
#             visualize(model.module, data, save=True, epoch=epoch+1)
#             torch.save(model.module.state_dict(), f'outputs/checkpoints/model_epoch_{epoch+1}.pth')
    
# model = MainModel()

# model = nn.DataParallel(model)

# pretrain_epochs = 75
# gan_epochs = 150
# total_epochs = gan_epochs


# train_model(
#     model=model,
#     train_dl=train_dl,
#     val_dl=val_dl,
#     epochs=total_epochs,
#     pretrain_epochs=pretrain_epochs,
#     display_every=5
# )

import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from utils import lab_to_rgb, AverageMeter, visualize
from dataset import train_dl, val_dl
from model import MainModel

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directories
os.makedirs("outputs/checkpoints", exist_ok=True)
os.makedirs("outputs/images", exist_ok=True)

def pretrain_generator(net_G, train_dl, criterion, optimizer, epochs):
    """Pretrain the Generator using L1 Loss"""
    net_G.train()
    for epoch in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            L = data['L'].to(device)
            ab = data['ab'].to(device)

            preds = net_G(L)
            loss = criterion(preds, ab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), L.size(0))

        print(f"Pretraining Epoch [{epoch+1}/{epochs}], Loss: {loss_meter.avg:.5f}")

def validate_model(model, val_dl):
    """Run validation and compute L1 loss"""
    model.eval()
    val_loss_meter = AverageMeter()
    criterion = nn.L1Loss()
    with torch.no_grad():
        for data in val_dl:
            L = data['L'].to(device)
            ab_real = data['ab'].to(device)
            ab_fake = model.module.net_G(L)
            loss = criterion(ab_fake, ab_real)
            val_loss_meter.update(loss.item(), L.size(0))
    return val_loss_meter.avg

def train_model(model, train_dl, val_dl, epochs, pretrain_epochs=5, display_every=5):
    """Train the model with pretraining + GAN training"""
    
    # Move model to device
    model.to(device)

    # Pretrain Generator
    print("Starting Generator Pretraining...")
    pretrain_generator(
        net_G=model.module.net_G,
        train_dl=train_dl,
        criterion=model.module.L1criterion,
        optimizer=model.module.opt_G,
        epochs=pretrain_epochs
    )
    print("Pretraining Completed.\n")

    best_val_loss = float("inf")  # Track best validation loss
    
    # Training with GAN
    for epoch in range(epochs):
        model.module.net_G.train()
        model.module.net_D.train()
        loss_meter_dict = {'loss_D': AverageMeter(), 'loss_G': AverageMeter()}

        for data in tqdm(train_dl):
            model.module.setup_input(data)
            model.module.optimize()

            # Update loss meters
            loss_meter_dict['loss_D'].update(model.module.loss_D.item(), data['L'].size(0))
            loss_meter_dict['loss_G'].update(model.module.loss_G.item(), data['L'].size(0))

        # Validation and Save Best Model
        val_loss = validate_model(model, val_dl)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.module.state_dict(), 'outputs/checkpoints/best_model.pth')
            print(f"✅ Saved best model with val_loss: {best_val_loss:.5f}")
        
        # Display & Save Model Every `display_every` Epochs
        if (epoch + 1) % display_every == 0:
            print(f"\n📢 Epoch [{epoch+1}/{epochs}]")
            print(f"Loss_D: {loss_meter_dict['loss_D'].avg:.5f}, Loss_G: {loss_meter_dict['loss_G'].avg:.5f}, Val_Loss: {val_loss:.5f}")
            data = next(iter(val_dl))
            visualize(model.module, data, save=True, epoch=epoch+1)
        
        torch.save(model.module.state_dict(), f'outputs/checkpoints/model_epoch_{epoch+1}.pth')

# Initialize and wrap model with DataParallel
model = MainModel()
model = nn.DataParallel(model)

pretrain_epochs = 75
gan_epochs = 150
total_epochs = gan_epochs

train_model(
    model=model,
    train_dl=train_dl,
    val_dl=val_dl,
    epochs=total_epochs,
    pretrain_epochs=pretrain_epochs,
    display_every=10
)
