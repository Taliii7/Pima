import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import tifffile
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import random

# ─── PARAMETRES ───────────────────────────────────────────────────────────────
IMAGE_PATH   = "puma_data_converted/test/images/training_set_metastatic_roi_027.tif"       # image CytoDArk0
MASK_PATH    = "puma_data_converted/test/masks/training_set_metastatic_roi_027_nuclei_mask.tiff"        # ground truth mask (instance)
SAM_CKPT     = "sam_vit_h_4b8939.pth"         # checkpoint SAM ViT-H
SAM_TYPE     = "vit_h"
OUTPUT_PATH  = "sam_zero_shot_comparison.png"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path):
    """Charge une image RGB en numpy array uint8."""
    img = tifffile.imread(path)
    if img.ndim == 3 and img.shape[0] <= 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] == 1:
        img = np.concatenate([img]*3, axis=-1)
    return img[:, :, :3]

def load_mask(path):
    """Charge le masque d'instances (valeurs entières, 0=fond)."""
    mask = tifffile.imread(path)
    if mask.ndim == 3:
        mask = mask[0]
    return mask.astype(np.int32)

def colorize_instances(instance_mask, black_background=True):
    """Colorie chaque instance avec une couleur aléatoire, fond noir."""
    h, w = instance_mask.shape
    if black_background:
        # fond noir opaque
        colored = np.zeros((h, w, 4), dtype=np.uint8)
        colored[:, :, 3] = 255  # opaque partout
    else:
        colored = np.zeros((h, w, 4), dtype=np.uint8)
    
    ids = np.unique(instance_mask)
    ids = ids[ids != 0]
    random.seed(42)
    for inst_id in ids:
        color = [random.randint(50, 255) for _ in range(3)] + [255]
        colored[instance_mask == inst_id] = color
    return colored

def save_comparison(image_rgb, gt_mask, pred_mask, output_path):
    """Sauvegarde la figure en 3 panneaux : image | GT | prédiction SAM."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1 : image originale
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Panel 2 : ground truth sur fond noir
    black_bg = np.zeros_like(image_rgb)
    axes[1].imshow(black_bg)
    gt_colored = colorize_instances(gt_mask, black_background=True)
    axes[1].imshow(gt_colored)
    n_gt = len(np.unique(gt_mask)) - 1
    axes[1].set_title(f"Ground Truth ({n_gt} cells)", 
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Panel 3 : prédiction SAM sur fond noir
    axes[2].imshow(black_bg)
    pred_colored = colorize_instances(pred_mask, black_background=True)
    axes[2].imshow(pred_colored)
    n_pred = len(np.unique(pred_mask)) - 1
    axes[2].set_title(f"SAM Zero-Shot ({n_pred} cells detected)", 
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("SAM Zero-Shot vs Ground Truth on PUMA", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure sauvegardée : {output_path}")

def masks_from_sam(masks_list, shape):
    """Convertit la liste de masques SAM en masque d'instances."""
    instance_mask = np.zeros(shape[:2], dtype=np.int32)
    # Trier par aire décroissante pour éviter les recouvrements
    masks_sorted = sorted(masks_list, key=lambda x: x['area'], reverse=True)
    for i, m in enumerate(masks_sorted, start=1):
        instance_mask[m['segmentation']] = i
    return instance_mask

def run_sam_inference(image_rgb, sam_ckpt, model_type, device):
    """Lance SAM en mode automatique (zero-shot)."""
    print(f"Chargement de SAM {model_type} sur {device}...")
    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.to(device)
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        min_mask_region_area=100,
    )
    print("Inference SAM en cours...")
    masks = generator.generate(image_rgb)
    print(f"  → {len(masks)} masques générés")
    return masks



# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Chargement
    image_rgb = load_image(IMAGE_PATH)
    gt_mask   = load_mask(MASK_PATH)
    print(f"Image: {image_rgb.shape}, GT instances: {len(np.unique(gt_mask))-1}")

    # Inference SAM
    sam_masks   = run_sam_inference(image_rgb, SAM_CKPT, SAM_TYPE, DEVICE)
    pred_mask   = masks_from_sam(sam_masks, image_rgb.shape)

    # Sauvegarde
    save_comparison(image_rgb, gt_mask, pred_mask, OUTPUT_PATH)