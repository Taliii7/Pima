import sys
import os
import argparse
import torch
import numpy as np
import tifffile
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, EnsureTyped, ScaleIntensityd, ScaleIntensityRangePercentilesd
)
from monai.data import Dataset, DataLoader
from monai.inferers import SlidingWindowInfererAdapt


dossier_actuel = os.path.dirname(os.path.abspath(__file__))
dossier_common = os.path.join(dossier_actuel, '../common')
if dossier_common not in sys.path:
    sys.path.append(dossier_common)

try:
    from evaluate_models import get_pq
    from utils import save_metrics_to_csv, deviceChoice 
except Exception as e:
    print(f"Erreur lors du chargement de common/ : {e}")
    sys.exit(1)

try:
    from components import LoadTiffd, LogitsToLabels
    from cell_sam_wrapper import CellSamWrapper
except ImportError:
    from scripts.components import LoadTiffd, LogitsToLabels
    from scripts.cell_sam_wrapper import CellSamWrapper

#  FONCTIONS UTILITAIRES VISTA
def load_vista_model(sam_ckpt, vista_ckpt, device):
    """Charge l'architecture VISTA-2D (SAM) et ses poids."""
    model = CellSamWrapper(checkpoint=sam_ckpt)
    model.to(device)
    
    checkpoint = torch.load(vista_ckpt, map_location=device, weights_only=True)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def get_preprocessing_transforms():
    """Reproduit les étapes de preprocessing exactes de VISTA."""
    return Compose([
        LoadTiffd(keys=["image"]),
        EnsureTyped(keys=["image"], data_type="tensor", dtype=torch.float),
        ScaleIntensityd(keys=["image"], minv=0, maxv=1, channel_wise=True),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, channel_wise=True, clip=True
        )
    ])


def EvaluateDataset(test_dir, device, model_path, sam_ckpt, iou_threshold=0.5):
    """Évalue le modèle VISTA-2D sur tout un dossier."""
    print(f"\n=== Lancement de l'évaluation VISTA-2D sur : {test_dir} ===")
    
    if not os.path.exists(test_dir):
        print(f"Erreur : Le dossier {test_dir} n'existe pas.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    # les chemins de gt et masks 
    dossier_img = os.path.join(test_dir, 'images')
    dossier_lbl = os.path.join(test_dir, 'masks')
    
    if not os.path.exists(dossier_img):
        dossier_img = test_dir
        dossier_lbl = test_dir

    chemins_images = sorted([os.path.join(dossier_img, f) for f in os.listdir(dossier_img) if f.endswith('.png') or f.endswith('.tif')])
    
    if len(chemins_images) == 0:
        print("Aucune image trouvée.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Chargement du modèle VISTA
    model = load_vista_model(sam_ckpt, model_path, device)
    transforms = get_preprocessing_transforms()
    
    # Paramètres de la fenêtre glissante
    inferer = SlidingWindowInfererAdapt(
        roi_size=[256, 256], sw_batch_size=4, overlap=0.625, 
        mode="gaussian", cache_roi_weight_map=True, progress=False
    )
    post_processor = LogitsToLabels()
    
    global_tp, global_fp, global_fn = 0, 0, 0
    list_dq, list_sq, list_pq = [], [], []

    # Accélération matérielle
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.float32

    # Boucle sur chaque image
    for img_path in tqdm(chemins_images, desc="Inférence VISTA"):
        
        # 2. RECHERCHE DU MASQUE CORRESPONDANT
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # On tente de trouver le nouveau format (_mask.png) ou l'ancien (_masks.tiff)
        gt_path_png    = os.path.join(dossier_lbl, f"{base_name}_nuclei_mask.png")
        gt_path_tiff   = os.path.join(dossier_lbl, f"{base_name}_nuclei_mask.tiff")
        gt_path_old    = os.path.join(dossier_lbl, f"{base_name}_masks.tiff")
        gt_path_old2   = os.path.join(dossier_lbl, f"{base_name}_mask.png")

        gt_path = None
        for candidate in [gt_path_tiff, gt_path_png, gt_path_old, gt_path_old2]:
            if os.path.exists(candidate):
                gt_path = candidate
                break

        if gt_path is None:
            continue
        
        # Forcer la lecture 16-bit correcte
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt_mask is None:
            continue
        if gt_mask.ndim > 2:
            gt_mask = gt_mask[:, :, 0]
        gt_mask = gt_mask.astype(np.int32)

        # Préparation MONAI de l'image
        data_dict = {"image": img_path}
        dataset = Dataset(data=[data_dict], transform=transforms)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Prédiction de MONAI
        for batch in dataloader:
            inputs = batch["image"].to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                    logits = inferer(inputs=inputs, network=model)
                
                logits_b0 = logits[0] 
                pred_mask, _ = post_processor(logits_b0, filename=img_path)
                
            break # On force l'arrêt après le premier batch
            
        # 4. SÉCURITÉ FORMAT PRÉDICTION
        # VISTA peut renvoyer un Tensor PyTorch, il faut le convertir en tableau NumPy pour get_pq
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
            
        # Si la prédiction a des dimensions supplémentaires (ex: [1, H, W]), on la met à plat
        if pred_mask.ndim > 2:
            pred_mask = pred_mask.squeeze()
            
        pred_mask = pred_mask.astype(np.int32)

        # 5. ÉVALUATION (Le Juge)
        pq_metrics, counts, _, _ = get_pq(gt_mask, pred_mask, match_iou=iou_threshold)
        
        dq, sq, pq = pq_metrics
        tp, fp, fn = counts

        list_dq.append(dq)
        list_sq.append(sq)
        list_pq.append(pq)
        global_tp += tp
        global_fp += fp
        global_fn += fn

    # Calcul final
    avg_dq = np.mean(list_dq) if list_dq else 0
    avg_sq = np.mean(list_sq) if list_sq else 0
    avg_pq = np.mean(list_pq) if list_pq else 0

    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n--- RÉSULTATS VISTA-2D ---")
    print(f" PQ: {avg_pq*100:.2f}% | F1: {f1_score*100:.2f}%")
    print(f" Total TP: {global_tp} | FP: {global_fp} | FN: {global_fn}")

    return len(list_dq), avg_sq, avg_dq, avg_pq, global_tp, global_fp, global_fn, precision, recall, f1_score


def evaluate_single_image_vista(image_path, gt_path, model_path, sam_ckpt, iou_threshold=0.5):
    """Évalue une seule image avec VISTA-2D et affiche les métriques."""
    device = deviceChoice() # Utilise ta fonction utilitaire pour MPS/CUDA
    
    print(f"\n[1/4] Chargement de l'image : {os.path.basename(image_path)}")
    
    # 1. Chargement du modèle et des transforms
    model = load_vista_model(sam_ckpt, model_path, device)
    transforms = get_preprocessing_transforms()
    
    # 2. Inférence VISTA (MONAI)
    print(f"[2/4] Segmentation VISTA-2D en cours...")
    data_dict = {"image": image_path}
    dataset = Dataset(data=[data_dict], transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1)
    
    inferer = SlidingWindowInfererAdapt(
        roi_size=[256, 256], sw_batch_size=1, overlap=0.625, 
        mode="gaussian", cache_roi_weight_map=True, progress=False
    )
    post_processor = LogitsToLabels()

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["image"].to(device)
            logits = inferer(inputs=inputs, network=model)
            logits_b0 = logits[0] 
            pred_mask, _ = post_processor(logits_b0, filename=image_path)
            break

    # Conversion sécurité pour la prédiction
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if pred_mask.ndim > 2:
        pred_mask = pred_mask.squeeze()
    pred_mask = pred_mask.astype(np.int32)

    # 3. Chargement Intelligent du Ground Truth
    print(f"[3/4] Calcul des métriques (IoU >= {iou_threshold})...")
    if os.path.exists(gt_path):
        # Chargement en 16-bit sans destruction des IDs
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        
        # Sécurité Rétrocompatibilité (si c'est un ancien format RGB)
        if gt_mask.ndim > 2:
            gt_mask = gt_mask[:, :, 0]
            
        gt_mask = gt_mask.astype(np.int32)

        # DEBUG : Affichage des infos pour être sûr
        print(f"     GT Max ID   : {np.max(gt_mask)} (Type: {gt_mask.dtype})")
        print(f"     Pred Max ID : {np.max(pred_mask)} (Type: {pred_mask.dtype})")

        # 4. Visualisation du Debug Overlap (Rouge vs Vert)
        plt.figure(figsize=(10, 10))
        plt.imshow(gt_mask > 0, cmap='Reds', alpha=0.5)
        plt.imshow(pred_mask > 0, cmap='Greens', alpha=0.5)
        plt.title(f"VISTA Debug: Rouge (GT) vs Vert (Pred) - {os.path.basename(image_path)}")
        plt.axis('off')
        plt.savefig('debug_overlap_vista.png', bbox_inches='tight')
        print("     Image de débogage générée : debug_overlap_vista.png")

        # 5. Calcul des métriques PQ
        pq_metrics, counts, _, _ = get_pq(gt_mask, pred_mask, match_iou=iou_threshold)
        dq, sq, pq = pq_metrics
        tp, fp, fn = counts

        print("\n" + "="*50)
        print(f" RÉSULTATS VISTA-2D : {os.path.basename(image_path)}")
        print("="*50)
        print(f" SQ (Segmentation Quality) : {sq*100:.2f}%")
        print(f" DQ (Detection Quality)    : {dq*100:.2f}%")
        print(f" PQ (Panoptic Quality)     : {pq*100:.2f}%")
        print("-" * 50)
        print(f" TP: {tp} | FP: {fp} | FN: {fn}")
        print("="*50)
    else:
        print(" [!] Ground Truth non trouvé au chemin indiqué.")


def main():
    parser = argparse.ArgumentParser(description="Évaluation VISTA-2D")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le modèle VISTA fine-tuné")
    parser.add_argument("--sam_ckpt", type=str, default="models/sam_vit_b_01ec64.pth", help="Poids SAM de base pour l'architecture")
    parser.add_argument("--iou", type=float, default=0.5)
    
    parser.add_argument("--famille", type=str, default="VISTA-2D")
    parser.add_argument("--modele_nom", type=str, required=True)
    parser.add_argument("--zoom", type=str, required=True)
    parser.add_argument("--dim", type=str, required=True)
    parser.add_argument("--csv", type=str, default="../resultats_comparaison/metriques_vista.csv")
    args = parser.parse_args()
    
    device =deviceChoice()
    
    nb_img, sq, dq, pq, tp, fp, fn, prec, rec, f1 = EvaluateDataset(
        test_dir=args.test_dir, device=device, model_path=args.model, 
        sam_ckpt=args.sam_ckpt, iou_threshold=args.iou
    )

    if nb_img > 0:
        save_metrics_to_csv(
            csv_path=args.csv, famille=args.famille, modele=args.modele_nom,
            zoom=args.zoom, dim=args.dim, iou=args.iou, nb_images=nb_img,
            sq=sq, dq=dq, pq=pq, tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1
        )

if __name__ == "__main__":
    main()
