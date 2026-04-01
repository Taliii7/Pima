import sys
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Imports MONAI & VISTA
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, EnsureTyped, ScaleIntensityd, ScaleIntensityRangePercentilesd,
    RandSpatialCropd, RandAxisFlipd, RandGaussianNoised, RandAdjustContrastd
)

# Gestion de l'import du parent pour cyrk0_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cyrk0_split import train_val_split
from scripts.components import LoadTiffd, LabelsToFlows, CellLoss
from scripts.cell_sam_wrapper import CellSamWrapper

def deviceChoice():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def get_transforms(roi_size=[256, 256], is_train=True):
    # On utilise les mêmes transforms de base
    xforms = [
        LoadTiffd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], data_type="tensor", dtype=torch.float),
        ScaleIntensityd(keys=["image"], minv=0, maxv=1, channel_wise=True),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, channel_wise=True, clip=True),
    ]
    if is_train:
        # Augmentations uniquement pour le train
        xforms.extend([
            RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
            RandAxisFlipd(keys=["image", "label"], prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.25, gamma=(1, 2)),
        ])
    
    xforms.append(LabelsToFlows(keys="label", flow_key="flow"))
    return Compose(xforms)

def plot_losses(args, train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='orange')
    plt.title(f"Suivi Overfitting : {args.output_name}")
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"Loss_curve_{args.output_name}.png")
    plt.close()

def run(args):
    device = deviceChoice()
    # On récupère Train ET Val
    t_img, t_lbl, v_img, v_lbl = train_val_split(base_dir=args.base_dir, zoom=args.zoom, dim=args.dim)
    
    train_files = [{"image": i, "label": l} for i, l in zip(t_img, t_lbl)]
    val_files = [{"image": i, "label": l} for i, l in zip(v_img, v_lbl)]

    train_loader = DataLoader(Dataset(train_files, get_transforms(is_train=True)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Dataset(val_files, get_transforms(is_train=False)), batch_size=1, shuffle=False)


    model = CellSamWrapper(checkpoint=args.sam_ckpt, auto_resize_inputs=True)
    model.load_state_dict(torch.load(args.vista_ckpt, map_location="cpu", weights_only=True), strict=False)
    model.to(device)
    
    for param in model.model.image_encoder.parameters():
        param.requires_grad = False
    print("Encodeur SAM gelé. Entraînement du décodeur uniquement !")

    loss_function = CellLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scaler = torch.amp.GradScaler('cuda')

    train_hist, val_hist = [], []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # --- PHASE ENTRAÎNEMENT ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Époque {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            inputs, targets = batch["image"].to(device), batch["flow"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(inputs)
                loss = loss_function(logits.float(), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- PHASE VALIDATION (Fin d'époque) ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_batch in tqdm(val_loader, desc="Validation"):
                v_inputs, v_targets = v_batch["image"].to(device), v_batch["flow"].to(device)
                with torch.amp.autocast('cuda'):
                    v_logits = model(v_inputs)
                    vl = loss_function(v_logits.float(), v_targets)
                val_loss += vl.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_hist.append(avg_train)
        val_hist.append(avg_val)
        
        print(f"Epoch {epoch+1}: Train={avg_train:.4f} | Val={avg_val:.4f}")

        # Sauvegarde du meilleur modèle
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"models/BEST_{args.output_name}.pt")
            print("⭐ Nouveau meilleur modèle sauvegardé !")

        torch.cuda.empty_cache()

    return train_hist, val_hist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="../cytoDArk_split")
    parser.add_argument("--output_name", type=str, default="vista_exp")
    parser.add_argument("--sam_ckpt", type=str, default="models/sam_vit_b_01ec64.pth")
    parser.add_argument("--vista_ckpt", type=str, default="models/model.pt")
    parser.add_argument("--zoom", type=int, default=0)
    parser.add_argument("--dim", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01) # NVIDIA default
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20) # Réduit pour éviter l'overfit
    args = parser.parse_args()

    t_loss, v_loss = run(args)
    plot_losses(args, t_loss, v_loss)

if __name__ == "__main__":
    main()
