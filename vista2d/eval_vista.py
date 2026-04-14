import sys
import os
import argparse
import torch
import numpy as np
import tifffile
from tqdm import tqdm

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
    
    chemins_images = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png') or f.endswith('.tif')])
    chemins_labels = [p.replace('.png', '_masks.tiff').replace('.tif', '_masks.tiff') for p in chemins_images]
    
    if len(chemins_images) == 0:
        print("Aucune image trouvée.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Chargement du modèle
    model = load_vista_model(sam_ckpt, model_path, device)
    transforms = get_preprocessing_transforms()
    
    # Paramètres de la fenêtre glissante
    inferer = SlidingWindowInfererAdapt(
        roi_size=[256, 256], sw_batch_size=1, overlap=0.625, 
        mode="gaussian", cache_roi_weight_map=True, progress=False
    )
    post_processor = LogitsToLabels()
    
    global_tp, global_fp, global_fn = 0, 0, 0
    list_dq, list_sq, list_pq = [], [], []

    # Accélération matérielle
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.float32

    #boucle sur chaque image
    for img_path, gt_path in tqdm(zip(chemins_images, chemins_labels), total=len(chemins_images), desc="Inférence VISTA"):
        if not os.path.exists(gt_path):
            continue
        
        # ground thruth
        gt_mask = tifffile.imread(gt_path)

        #preparation MONAI de l'image
        data_dict = {"image": img_path}
        dataset = Dataset(data=[data_dict], transform=transforms)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # prrdiction de monai
        for batch in dataloader:
            inputs = batch["image"].to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                    logits = inferer(inputs=inputs, network=model)
                
                logits_b0 = logits[0] 
                pred_mask, _ = post_processor(logits_b0, filename=img_path)
                
            break # on force l'arrêt après le premier batch (puisqu'on n'a qu'une image par itération)

        # scores
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

    return len(list_dq), avg_sq, avg_dq, avg_pq, global_tp, global_fp, global_fn, precision, recall, f1_score


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