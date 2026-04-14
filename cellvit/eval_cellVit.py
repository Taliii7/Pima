import sys
import os
import argparse
import torch
import tifffile
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image

# on cree un lien vers le repertoire commun aux méthodes essentiels
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

# importation spécifique à CellViT
try:
    from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
except ImportError as e:
    print(f"Erreur : Impossible d'importer CellViTSAM. {e}")
    sys.exit(1)


def load_model(checkpoint, device):
    """Charge le modèle CellVit avec son fichier de configuration intégré"""
    print(f"Chargement du modèle CellVit : {checkpoint}")
    checkpoint_info = torch.load(checkpoint, map_location=device)
    conf = checkpoint_info['config']
    
    model = CellViTSAM(
        model_path=None,
        num_nuclei_classes=conf["data.num_nuclei_classes"],
        num_tissue_classes=conf["data.num_tissue_classes"],
        vit_structure='SAM-H',
        drop_rate=conf['training.drop_rate']
    )

    model.load_state_dict(checkpoint_info['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def load_normalize_x(img_path, device):
    """Lecture et normalisation de l'image pour CellVit"""
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return transform(img).unsqueeze(0).to(device)


def EvaluateDataset(test_dir, device, model_path, zoom_str, iou_threshold=0.5):
    """Évalue CellVit sur un dossier et calcule le SQ, DQ, PQ"""
    print(f"\n=== Lancement de l'évaluation CellVit sur : {test_dir} ===")
    
    if not os.path.exists(test_dir):
        print(f"Erreur : Le dossier {test_dir} n'existe pas.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    chemins_images = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')])
    chemins_labels = [p.replace('.png', '_masks.tiff') for p in chemins_images]
    
    if len(chemins_images) == 0:
        print("Aucune image trouvée dans ce dossier.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    mag = 40
    if "20x" in zoom_str:
        mag = 20

    model = load_model(model_path, device)
    
    global_tp, global_fp, global_fn = 0, 0, 0
    list_dq, list_sq, list_pq = [], [], []

#boucle d'évaluation
    for img_path, gt_path in tqdm(zip(chemins_images, chemins_labels), total=len(chemins_images), desc="Inférence CellVit"):
        if not os.path.exists(gt_path):
            tqdm.write(f"Masque Ground Truth introuvable pour {os.path.basename(img_path)}")
            continue
        
        #prediction de cellVit
        img_tensor = load_normalize_x(img_path, device)
        with torch.no_grad():
            outputs = model(img_tensor)
            instance_map, _ = model.calculate_instance_map(outputs, magnification=mag)
        
        #extraction du masque numpy
        if isinstance(instance_map[0], torch.Tensor):
            pred_mask = instance_map[0].cpu().numpy()
        else:
            pred_mask = instance_map[0]

        #lecture gt + calculs metriques
        gt_mask = tifffile.imread(gt_path)

        pq_metrics, counts, _, _ = get_pq(gt_mask, pred_mask, match_iou=iou_threshold)
        
        dq, sq, pq = pq_metrics
        tp, fp, fn = counts

        list_dq.append(dq)
        list_sq.append(sq)
        list_pq.append(pq)
        global_tp += tp
        global_fp += fp
        global_fn += fn

    # métriques finales
    avg_dq = np.mean(list_dq) if list_dq else 0
    avg_sq = np.mean(list_sq) if list_sq else 0
    avg_pq = np.mean(list_pq) if list_pq else 0

    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n--- RÉSULTATS CELLVIT ---")
    print("\n" + "="*60)
    print("RÉSULTATS DE L'ÉVALUATION (Panoptic Quality & COCO) ")
    print("="*60)
    print(f" Modèle évalué     : {os.path.basename(model_path) if model_path else 'Cellpose Standard'}")
    print(f" Seuil de rigueur  : IoU >= {iou_threshold}")
    print(f" Images traitées   : {len(list_dq)}")
    print("-" * 60)
    print("  QUALITÉ PANOPTIQUE (Moyenne par image)")
    print(f"   SQ (Segmentation Quality) : {avg_sq*100:05.2f}% (Précision des contours)")
    print(f"   DQ (Detection Quality)    : {avg_dq*100:05.2f}% (F1-score moyen par image)")
    print(f"   PQ (Panoptic Quality)     : {avg_pq*100:05.2f}% (Le score roi : SQ x DQ)")
    print("-" * 60)
    print(" DÉTECTION GLOBALE COCO (Toutes images confondues)")
    print(f"   Vrais Positifs (TP) : {global_tp} cellules correctement trouvées")
    print(f"   Faux Positifs  (FP) : {global_fp} fausses alarmes (bruit/fond)")
    print(f"   Faux Négatifs  (FN) : {global_fn} vraies cellules ratées")
    print(f"   Précision           : {precision*100:05.2f}%")
    print(f"   Rappel (Recall)     : {recall*100:05.2f}%")
    print(f"   F1-Score Global     : {f1_score*100:05.2f}%")
    print("="*60 + "\n")

    return len(list_dq), avg_sq, avg_dq, avg_pq, global_tp, global_fp, global_fn, precision, recall, f1_score


def main():
    parser = argparse.ArgumentParser(description="Évaluation du modèle CellVit")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    
    parser.add_argument("--famille", type=str, default="CellVit")
    parser.add_argument("--modele_nom", type=str, required=True)
    parser.add_argument("--zoom", type=str, required=True)
    parser.add_argument("--dim", type=str, required=True)
    parser.add_argument("--csv", type=str, default="../resultats_comparaison/metriques_globales.csv")
    args = parser.parse_args()
    
    device = deviceChoice()
    
    nb_img, sq, dq, pq, tp, fp, fn, prec, rec, f1 = EvaluateDataset(
        test_dir=args.test_dir, device=device, model_path=args.model, 
        zoom_str=args.zoom, iou_threshold=args.iou
    )

    if nb_img > 0:
        save_metrics_to_csv(
            csv_path=args.csv, famille=args.famille, modele=args.modele_nom,
            zoom=args.zoom, dim=args.dim, iou=args.iou, nb_images=nb_img,
            sq=sq, dq=dq, pq=pq, tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1
        )

if __name__ == "__main__":
    main()