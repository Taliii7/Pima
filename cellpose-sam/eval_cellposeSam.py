import sys
import torch
import argparse
import warnings
import cv2

from cellpose import models, io, plot, metrics
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from eval_cellposeSam import evaluate_single_image

current_dir = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_dir, '..', 'common')

if common_path not in sys.path:
    sys.path.append(common_path)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importation de TA méthode de calcul (Panoptic Quality)
try:
    from evaluate_models import get_pq
except ImportError:
    print("Erreur : Impossible d'importer get_pq. Vérifie tes dossiers 'common'.")

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from cyrk0_split import train_val_split, directory_split

def deviceChoice():
    # On prend le meilleur device possible, cuda (gpu de la fac) > mps (gpu metal de nos mac) > cpu 
    if torch.cuda.is_available():
        print("Super ! Le GPU NVIDIA du serveur est activé !")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Le GPU du Mac M1 (MPS) est activé.")
        device = torch.device("mps")
    else:
        print("Attention, on tourne sur le CPU.")
        device = torch.device("cpu")
    return device

def getImage(chemin_image): 
    return io.imread(chemin_image)

def SegmentOneImage(chemin_image,device,model_path=None):
    # charge l'image
    print(f"Chargement de l'image : {chemin_image}...")
    img = getImage(chemin_image)

    print(f"Image de dimension : {img.shape}. L'image a en dimension de channels  {img.shape[-1]} ")

    if model_path:
        # On dit à Cellpose de charger nos poids d'entraînement via "pretrained_model"
        print(f"Chargement de TON modèle personnalisé : {model_path}")
        model = models.CellposeModel(gpu=True, pretrained_model=model_path, device=device)
    else:
        # On charge le modèle par défaut de Cellpose
        print("Chargement du modèle Cellpose par défaut")
        model = models.CellposeModel(gpu=True, device=device)

    # lance la prédiction 
    print("Segmentation en cours...")
    masks, flows, styles = model.eval(img, diameter=None)

    nombre_cellules = masks.max()
    print(f" Terminé ! {nombre_cellules} cellules trouvées.")

    # sauvegarde le résultat
    fig = plt.figure(figsize=(12, 5))
    img_rgb = img[:, :, :3]
    print(img_rgb.shape," ",img_rgb[-1].shape," ", img_rgb.shape[-1])
    plot.show_segmentation(fig, img_rgb, masks, flows[0])
    plt.tight_layout()
    plt.savefig("resultat_serveur.png", dpi=300)
    print("Image sauvegardée sous 'resultat_serveur.png'")
    return  masks, flows ,styles

# méthode utilitaire pour sauvegarder le resultat de la segmentation à la racine
def showResult(img,masks,flows,output_name="resultat_serveur.png"):
    fig = plt.figure(figsize=(12, 5))
    img_plot = img[:, :, :3] if img.ndim == 3 and img.shape[-1] > 3 else img
    plot.show_segmentation(fig, img_plot, masks, flows[0])
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"Image sauvegardée sous '{output_name}'")

# --------------- évaluations des perfs ---------------

def evaluate_segmentation(masks, chemin_gt,baseline_black=False):
    # charge le ground truth qui est en .tiff
    print(f"\nChargement du ground truth : {chemin_gt}...")
    gt_image = getImage(chemin_gt)

    if len(gt_image.shape) > 2:
        gt_image = gt_image[:, :, 0] 
    
    pred_binary = masks > 0
    gt_binary = gt_image > 0

    black_baseline = np.zeros_like(gt_binary)
    evaluate_metrics(gt_binary, pred_binary, "Cellpose-SAM")
    evaluate_metrics(gt_binary, black_baseline, "Baseline (Tout Noir)")

#fonction pour calculer les métriques
def evaluate_metrics(y_true, y_pred, name="Modèle"):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten() 

    iou= jaccard_score(y_true_f, y_pred_f, zero_division=0)
    dice = f1_score(y_true_f, y_pred_f, zero_division=0)
    precision = precision_score(y_true_f, y_pred_f, zero_division=0)
    recall = recall_score(y_true_f, y_pred_f, zero_division=0)
    
    print(f"\n--- Performances : {name} ---")
    print(f"IoU      : {iou:.4f}  (IoU et jaccard sont les mêmes trucs) (Score d'overlap strict)")
    print(f"Dice     : {dice:.4f} (la même chose que le F1-Score)")
    print(f"Precision: {precision:.4f} (Parmi ce qui est prédit, combien est correct ?)")
    print(f"Recall   : {recall:.4f} (Parmi les vraies cellules, combien ont été trouvées ?)")

def print_coco_metrics(tp, fp, fn, seuils):
    print("   $$$$. $. $$$$$$$$$$$$.  PERFORMANCES GLOBALES (Norme COCO / Par Instance).     $$$$$$$$$$$$$$$$$$$$$$$$    ")
    for i, seuil in enumerate(seuils):
        total_tp = np.sum(tp[:, i]) 
        total_fp = np.sum(fp[:, i]) 
        total_fn = np.sum(fn[:, i]) 
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n[ Seuil de rigueur IoU = {seuil} ]")
        print(f"  Vrais Positifs (Cellules OK)     : {total_tp}")
        print(f"  Faux Positifs  (Fausses alarmes) : {total_fp}")
        print(f"  Faux Négatifs  (Cellules ratées) : {total_fn}")
        print(f"\nPrécision : {precision*100:.2f}%")
        print(f"Rappel    : {recall*100:.2f}%")
        print(f"F1-Score  : {f1_score*100:.2f}%")

def test(image_path,label_path,device, gpu=True, model_path=None):
    test_data =getImage(image_path)
    test_labels =getImage(label_path)
    
    if model_path:
        model = models.CellposeModel(gpu=True, pretrained_model=model_path, device=device)
    else:
        model = models.CellposeModel(gpu=True, device=device)
        
    print(f"Prédiction  sur {image_path} images en cours...")
    masks_pred, flows, styles = model.eval(test_data, diameter=None)
    
    seuils_iou = [0.5, 0.75, 0.9]
    ap, tp, fp, fn = metrics.average_precision(test_labels, masks_pred, threshold=seuils_iou)
    print_coco_metrics(tp, fp, fn, seuils_iou)

# =========================================================================
# RECHERCHE AUTOMATIQUE DES EDGE CASES (Basé sur la Panoptic Quality)
# =========================================================================
def find_edge_cases(dossier_racine, device, model_path=None, iou_threshold=0.5):
    """
    Parcourt une arborescence complexe, filtre UNIQUEMENT les dossiers 'test',
    évalue le modèle sur chaque image avec la Panoptic Quality (get_pq),
    et affiche le Top 5 et le Flop 5 (Edge Cases) basés sur la PQ.
    """
    print(f"\n=== 🕵️‍♂️ Recherche Globale des Edge Cases (via Panoptic Quality) dans : {dossier_racine} ===")
    
    chemins_images = []
    
    # 1. Parcourir récursivement tous les sous-dossiers avec os.walk
    for root, dirs, files in os.walk(dossier_racine):
        if os.path.basename(root) != 'test':
            continue
            
        for f in files:
            if (f.endswith('.png') or f.endswith('.tif') or f.endswith('.tiff')) and 'mask' not in f.lower():
                chemins_images.append(os.path.join(root, f))
                
    chemins_images = sorted(chemins_images)
    print(f"-> {len(chemins_images)} images de TEST trouvées au total !")

    if len(chemins_images) == 0:
        print("❌ Aucune image trouvée.")
        return

    # 2. Charger le modèle
    if model_path and os.path.exists(model_path):
        print(f"Chargement de ton modèle : {model_path}")
        model = models.CellposeModel(gpu=(device.type != 'cpu'), pretrained_model=model_path, device=device)
    else:
        print("Chargement du modèle de base...")
        model = models.CellposeModel(gpu=(device.type != 'cpu'), device=device)

    resultats_images = []

    # 3. Boucler sur chaque image
    for img_path in tqdm(chemins_images, desc="Analyse des Edge Cases", unit="img"):
        
        base_name = os.path.splitext(img_path)[0]
        
        # Recherche du fichier masque
        gt_path = None
        for ext in ['_nuclei_mask.png', '_mask.png', '_masks.png', '_masks.tiff', '_masks.tif', '_mask.tif']:
            if os.path.exists(base_name + ext):
                gt_path = base_name + ext
                break 
                
        if not gt_path:
            tqdm.write(f"[Attention] Ground truth introuvable pour : {img_path}")
            continue
        
        # Chargement de l'image (format Cellpose)
        img = io.imread(img_path)
        if img.shape[-1] > 3:
            img = img[..., :3]

        # Chargement intelligent du GT (16-bit support)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt_mask is None:
            continue
            
        if gt_mask.ndim > 2:
            gt_mask = gt_mask[:, :, 0]
            
        gt_mask = gt_mask.astype(np.int32)
        
        # Inférence
        masks_pred, _, _ = model.eval(img, diameter=None)
        masks_pred = masks_pred.astype(np.int32)

        # Sécurité de redimensionnement
        if masks_pred.shape != gt_mask.shape:
            masks_pred = cv2.resize(masks_pred.astype(np.float32), 
                                    (gt_mask.shape[1], gt_mask.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST).astype(np.int32)

        # --- LE CŒUR DU CALCUL : Utilisation de TA fonction get_pq ---
        pq_metrics, counts, _, _ = get_pq(gt_mask, masks_pred, match_iou=iou_threshold)
        dq, sq, pq = pq_metrics
        tp, fp, fn = counts
        
        # Formatage du nom pour l'affichage
        parts = img_path.split(os.sep)
        nom_image_propre = f"{parts[-4]}/{parts[-3]}/{parts[-1]}" if len(parts) >= 4 else os.path.basename(img_path)
        
        # On sauvegarde toutes les métriques
        resultats_images.append((nom_image_propre, pq, sq, dq, tp, fp, fn, img_path))
        
    # 4. Trier les résultats par PQ (Panoptic Quality)
    resultats_images.sort(key=lambda x: x[1])
    
    # 5. Affichage des résultats
    if len(resultats_images) > 0:
        print("\n🚨 --- EDGE CASES (Pires images classées par Panoptic Quality) --- 🚨")
        for name, pq, sq, dq, tp_v, fp_v, fn_v, full_path in resultats_images[:5]:
            print(f"[{name}] PQ: {pq*100:05.2f}% (SQ: {sq*100:05.2f}%, DQ: {dq*100:05.2f}%) | TP: {tp_v}, FP: {fp_v}, FN: {fn_v}")
            print(f"    -> {full_path}")
            
        print("\n✅ --- CAS PARFAITS (Meilleures images classées par Panoptic Quality) --- ✅")
        for name, pq, sq, dq, tp_v, fp_v, fn_v, full_path in resultats_images[-5:]:
            print(f"[{name}] PQ: {pq*100:05.2f}% (SQ: {sq*100:05.2f}%, DQ: {dq*100:05.2f}%) | TP: {tp_v}, FP: {fp_v}, FN: {fn_v}")
            print(f"    -> {full_path}")
    else:
        print("\n❌ Aucun résultat généré.")

def main(): 
    parser = argparse.ArgumentParser(description="Script pour évaluer CellposeSAM et trouver les Edge Cases")

    #entrée / sortie
    parser.add_argument("--base_dir", type=str, default="cytoDArk_split", help="Le dossier racine")
    parser.add_argument("--image", type=str, default="../cytoDArk_split/20x/256x256/test/ID5_Aud_Cortex_Tursiops_1.png", help="L'image ")
    parser.add_argument("--gt", type=str, default="../cytoDArk_split/20x/256x256/test/ID5_Aud_Cortex_Tursiops_1_masks.tiff", help="Le ground truth ")
    parser.add_argument("--out_dir", type=str, default="../output", help="Le nom du dossier de sortie")
    parser.add_argument("--output_name", type=str, default="../cellpose_dauphin_complet", help="Le nom de fichier de sortie")
    parser.add_argument("--model", type=str, default="models/cellpose_dauphin_complet", help="Le lien du model")
    
    # AJOUT DE L'ARGUMENT POUR LES EDGE CASES
    parser.add_argument("--find_edges", action="store_true", help="Cherche les pires/meilleures images d'un dossier")

    args = parser.parse_args()

    device = deviceChoice() 
    modele_perso = args.model

    # GESTION DU MODE EDGE CASES
    if args.find_edges and args.base_dir:
        print("\n--- Mode : Recherche des Edge Cases ---")
        find_edge_cases(dossier_racine=args.base_dir, device=device, model_path=modele_perso)

    elif (args.image and args.gt):
        print("\n--- TEST VISUEL SUR UNE IMAGE ---")
        chemin_image = args.image
        chemin_gt = args.gt
        
        evaluate_single_image(chemin_image,chemin_gt,model_path=modele_perso,iou_threshold=0.5 )

    elif (args.base_dir): 
        print("\n--- Analyse performance du model sur un dataset ---")
        dossier_test = args.base_dir

if __name__ == "__main__":
    main()