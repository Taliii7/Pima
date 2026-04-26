import os
import sys
import argparse
import torch
import numpy as np
import tifffile
import cv2

try:
    from evaluate_models import get_pq
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../common'))
    from evaluate_models import get_pq

from monai.transforms import (
    Compose, EnsureTyped, ScaleIntensityd, ScaleIntensityRangePercentilesd, Lambdad
)
from monai.data import Dataset, DataLoader
from monai.inferers import SlidingWindowInfererAdapt

try:
    from components import LogitsToLabels
    from cell_sam_wrapper import CellSamWrapper
except ImportError:
    from scripts.components import LogitsToLabels
    from scripts.cell_sam_wrapper import CellSamWrapper


def deviceChoice():
    if torch.cuda.is_available():
        print("Super ! Le GPU NVIDIA est activé !")
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        print("Le GPU du Mac M1/M2/M3 (MPS) est activé.")
        return torch.device("mps")
    else:
        print("Attention, on tourne sur le CPU.")
        return torch.device("cpu")


def get_optimal_roi_size(image_shape):
    """
    Choisit automatiquement le roi_size et overlap selon la taille de l'image.
    image_shape : (H, W) de l'image
    """
    h, w = image_shape
    max_dim = max(h, w)

    if max_dim <= 256:
        # Petites images (72x72, 256x256) -> un seul patch
        return [max_dim, max_dim], 0.0
    elif max_dim <= 1024:
        # Images moyennes (1024x1024) -> patches 256 avec overlap raisonnable
        return [256, 256], 0.25
    else:
        # Grandes images (2048x2048) -> patches 512
        return [512, 512], 0.25

def load_vista_model(sam_ckpt, vista_ckpt, device):
    print("Construction de l'architecture VISTA-2D (SAM)...")
    model = CellSamWrapper(checkpoint=sam_ckpt)
    model.to(device)
    print(f"Chargement des poids VISTA depuis : {vista_ckpt}")
    checkpoint = torch.load(vista_ckpt, map_location=device, weights_only=True)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_image_as_rgb_chw(filepath):
    """
    Charge une image (TIFF ou PNG/JPG) et retourne (3, H, W) float32.
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.tif', '.tiff']:
        img = tifffile.imread(filepath)
    else:
        # PNG, JPG, BMP... -> OpenCV (BGR -> RGB)
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] in [3, 4]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    
    print(f"[DEBUG load] shape brute : {img.shape}, dtype: {img.dtype}")

    # Normalisation float32
    if img.dtype == np.uint16:
        img = (img / 65535.0 * 255.0).astype(np.float32)
    else:
        img = img.astype(np.float32)

    # Grayscale (H, W) -> (H, W, 1)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    # HWC : garder 3 canaux
    if img.shape[2] >= 3:
        img = img[:, :, :3]
    elif img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))  # (3, H, W)

    print(f"[DEBUG load] shape finale CHW : {img.shape}")
    assert img.shape[0] == 3, f"Erreur canal : {img.shape}"
    return img


def load_image_monai(filepath):
    """Transform MONAI qui charge l'image via notre loader propre."""
    def _load(d):
        d["image"] = load_image_as_rgb_chw(d["image"])
        return d
    return _load


def get_preprocessing_transforms():
    """Pipeline propre sans LoadTiffd."""
    return Compose([
        # Étape 1 : Chargement propre par nous (remplace LoadTiffd)
        Lambdad(keys=["image"], func=lambda path: load_image_as_rgb_chw(path)),

        # Étape 2 : Conversion en tensor float
        EnsureTyped(keys=["image"], data_type="tensor", dtype=torch.float),

        # Étape 3 : Normalisation robuste
        ScaleIntensityd(keys=["image"], minv=0, maxv=1, channel_wise=True),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0,
            channel_wise=True, clip=True
        ),
    ])


def main():
    parser = argparse.ArgumentParser(description="Inférence VISTA-2D et Évaluation")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out", type=str, default="resultat_vista.tif")
    parser.add_argument("--sam_ckpt", type=str, default="models/sam_vit_b_01ec64.pth")
    parser.add_argument("--vista_ckpt", type=str, default="models/model.pt")
    parser.add_argument("--gt", type=str, default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    device = deviceChoice()
    model = load_vista_model(args.sam_ckpt, args.vista_ckpt, device)

    print(f"Chargement et prétraitement de l'image : {args.image}")
    transforms = get_preprocessing_transforms()
    dataset = Dataset(data=[{"image": args.image}], transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    inferer = SlidingWindowInfererAdapt(
        roi_size=[256, 256], 
        sw_batch_size=1, #changer ça à + plus tard sur gpu
        overlap=0.625,
        mode="gaussian", cache_roi_weight_map=True, progress=True  # progress=True pour voir l'avancement
    )

    print("Segmentation en cours (Sliding Window)...")
    post_processor = LogitsToLabels()

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["image"].to(device)
            print(f"[DEBUG] inputs shape : {inputs.shape}")  # Doit être [1, 3, 1024, 1024]

            # AMP désactivé sur MPS (instable), activé uniquement sur CUDA
            use_amp = device.type == "cuda"
            amp_dtype = torch.float16 if use_amp else torch.float32

            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                logits = inferer(inputs=inputs, network=model)

            print(f"[DEBUG] logits shape : {logits.shape}")  # Doit être [1, 3, 1024, 1024]

            logits_b0 = logits[0]
            pred_mask, _ = post_processor(logits_b0, filename=args.image)

            if isinstance(pred_mask, torch.Tensor):
                pred_mask = pred_mask.cpu().numpy()

            pred_mask = np.squeeze(pred_mask)
            assert pred_mask.ndim == 2, f"[ERREUR] pred_mask mal formé : {pred_mask.shape}"
            pred_mask = pred_mask.astype(np.int32)

            print(f"Terminé ! {pred_mask.max()} cellules trouvées.")
            tifffile.imwrite(args.out, pred_mask.astype(np.uint16))
            print(f"Masque sauvegardé sous : {args.out}")
            break

    # Évaluation
    if args.gt:
        print(f"\n[Évaluation] GT : {os.path.basename(args.gt)}")
        if not os.path.exists(args.gt):
            print("[!] Fichier GT introuvable.")
            return

        gt_mask = cv2.imread(args.gt, cv2.IMREAD_UNCHANGED)
        if gt_mask is not None and gt_mask.ndim > 2:
            gt_mask = gt_mask[:, :, 0]
        gt_mask = gt_mask.astype(np.int32)

        if pred_mask.shape != gt_mask.shape:
            print(f"[!] Correction taille : {pred_mask.shape} -> {gt_mask.shape}")
            pred_mask = cv2.resize(
                pred_mask.astype(np.float32),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        pq_metrics, counts, _, _ = get_pq(gt_mask, pred_mask, match_iou=args.iou)
        dq, sq, pq = pq_metrics
        tp, fp, fn = counts

        print("\n" + "="*50)
        print(f" RÉSULTATS VISTA-2D : {os.path.basename(args.image)}")
        print("="*50)
        print(f" SQ (Segmentation Quality) : {sq*100:.2f}%")
        print(f" DQ (Detection Quality)    : {dq*100:.2f}%")
        print(f" PQ (Panoptic Quality)     : {pq*100:.2f}%")
        print("-"*50)
        print(f" TP: {tp} | FP: {fp} | FN: {fn}")
        print("="*50)


if __name__ == "__main__":
    main()