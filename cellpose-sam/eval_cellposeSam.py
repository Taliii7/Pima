import sys
import os
import argparse
import torch
import numpy as np
from cellpose import models, io
from tqdm import tqdm

#On ajoute le dossier 'common' au chemin Python pour pouvoir importer evaluate_models.py et utils.py qui s'y trouvent
dossier_actuel = os.path.dirname(__file__)
dossier_common = os.path.abspath(os.path.join(dossier_actuel, '../common'))
sys.path.append(dossier_common)

try:
    from evaluate_models import get_pq
    from utils import deviceChoice
except ImportError:
    print(f"Erreur : Impossible de trouver 'evaluate_models.py' dans le dossier {dossier_common}")
    sys.exit(1)




def EvaluateDataset(test_dir, device, model_path=None, iou_threshold=0.5):
    """
    Évalue le modèle sur tout un dossier en utilisant evaluate_models.py
    """
    print(f"\n=== Lancement de l'évaluation sur le dossier : {test_dir} ===")
    
    if not os.path.exists(test_dir):
        print(f"Erreur : Le dossier {test_dir} n'existe pas.")
        return
    
    # 1. Lister les images et les labels (ground truth)
    chemins_images = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')])
    chemins_labels = [p.replace('.png', '_masks.tiff') for p in chemins_images]
    
    if len(chemins_images) == 0:
        print("Aucune image .png trouvée dans ce dossier.")
        return

    # 2. Charger le modèle
    if model_path and os.path.exists(model_path):
        print(f"Chargement de TON modèle personnalisé : {model_path}")
        model = models.CellposeModel(gpu=True, pretrained_model=model_path, device=device)
    else:
        print("Chargement du modèle Cellpose par défaut (Modèle personnalisé non trouvé ou non spécifié).")
        model = models.CellposeModel(gpu=True, device=device)

    # 3. Initialiser les compteurs globaux
    global_tp = 0
    global_fp = 0
    global_fn = 0
    list_dq, list_sq, list_pq = [], [], []

    print(f"Évaluation image par image en cours ({len(chemins_images)} images)...")
    
    # 4. Boucle d'évaluation
    for img_path, gt_path in tqdm(zip(chemins_images, chemins_labels), total=len(chemins_images), desc="Inférence et Scores", unit="img"):
        if not os.path.exists(gt_path):
            print(f"  [!] Masque Ground Truth introuvable pour {os.path.basename(img_path)} - Ignoré")
            continue
        
        # Lecture des images
        img = io.imread(img_path)
        gt_mask = io.imread(gt_path)

        # Prédiction de Cellpose-SAM
        masks_pred, flows, styles = model.eval(img, diameter=None)

        # APPEL AU FICHIER evaluate_models.py (La base commune)
        # get_pq retourne : ([dq, sq, pq], [tp, fp, fn], [pairs...], sum_iou)
        pq_metrics, counts, _, _ = get_pq(gt_mask, masks_pred, match_iou=iou_threshold)
        
        dq, sq, pq = pq_metrics
        tp, fp, fn = counts

        # Sauvegarde des scores de l'image
        list_dq.append(dq)
        list_sq.append(sq)
        list_pq.append(pq)
        
        # Cumul pour les scores globaux du dataset
        global_tp += tp
        global_fp += fp
        global_fn += fn

    # 5. Calcul des métriques finales
    avg_dq = np.mean(list_dq) if list_dq else 0
    avg_sq = np.mean(list_sq) if list_sq else 0
    avg_pq = np.mean(list_pq) if list_pq else 0

    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 6. Affichage du rapport final
    print("\n" + "="*60)
    print("RÉSULTATS DE L'ÉVALUATION (Panoptic Quality & COCO) ")
    print("="*60)
    print(f" Modèle évalué     : {os.path.basename(model_path) if model_path else 'Cellpose Standard'}")
    print(f" Seuil de rigueur  : IoU >= {iou_threshold}")
    print(f" Images traitées   : {len(list_dq)}")
    print("-" * 60)
    print(" 1 QUALITÉ PANOPTIQUE (Moyenne par image)")
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


def main():
    parser = argparse.ArgumentParser(description="Évaluation globale d'un modèle Cellpose-SAM sur un dossier")

    # Arguments de chemins
    parser.add_argument("--test_dir", type=str, 
                        default="../cytoDArk_split/20x/256x256/test", 
                        help="Chemin vers le dossier contenant les images de TEST")
    
    parser.add_argument("--model", type=str, 
                        default="models/cellpose_dauphin_complet", 
                        help="Chemin vers les poids de ton modèle fine-tuné")
    
    parser.add_argument("--iou", type=float, 
                        default=0.5, 
                        help="Seuil d'Intersection-over-Union pour valider une détection (défaut: 0.5)")

    args = parser.parse_args()
    device = deviceChoice()
    
    # Lancement de l'évaluation
    EvaluateDataset(test_dir=args.test_dir, device=device, model_path=args.model, iou_threshold=args.iou)


if __name__ == "__main__":
    main()