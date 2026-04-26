import sys
import os
import argparse
import torch
import numpy as np
from cellpose import models, io,plot
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

#On ajoute le dossier 'common' au chemin Python pour pouvoir importer evaluate_models.py et utils.py qui s'y trouvent
dossier_actuel = os.path.dirname(__file__)
dossier_common = os.path.abspath(os.path.join(dossier_actuel, '../common'))
sys.path.append(dossier_common)

try:
    from evaluate_models import get_pq
    from utils import save_metrics_to_csv, deviceChoice 
except ImportError:
    print(f"Erreur : Impossible de trouver 'evaluate_models.py' dans le dossier {dossier_common}")
    sys.exit(1)


def EvaluateDataset(test_dir, device, model_path=None, iou_threshold=0.5):
    """
    Évalue le modèle sur tout un dossier en utilisant evaluate_models.py
    (Version sécurisée 16-bit & Rétrocompatible)
    """
    print(f"\n=== Lancement de l'évaluation sur le dossier : {test_dir} ===")
    
    if not os.path.exists(test_dir):
        print(f"Erreur : Le dossier {test_dir} n'existe pas.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    # 1. Gestion intelligente des chemins (Support ancien/nouveau dataset)
    dossier_img = os.path.join(test_dir, 'images')
    dossier_lbl = os.path.join(test_dir, 'masks')
    
    if not os.path.exists(dossier_img):
        dossier_img = test_dir
        dossier_lbl = test_dir

    # On cherche les .png ET les .tif
    chemins_images = sorted([os.path.join(dossier_img, f) for f in os.listdir(dossier_img) if f.endswith(('.png', '.tif'))])
    
    if len(chemins_images) == 0:
        print("Aucune image trouvée dans ce dossier.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    #. Charger le modèle
    is_gpu = (device.type != 'cpu')
    if model_path and os.path.exists(model_path):
        print(f"Chargement de TON modèle personnalisé : {model_path}")
        model = models.CellposeModel(gpu=is_gpu, pretrained_model=model_path, device=device)
    else:
        print("Chargement du modèle Cellpose par défaut (Modèle personnalisé non trouvé ou non spécifié).")
        model = models.CellposeModel(gpu=is_gpu, device=device)

    # Initialiser les compteurs globaux
    global_tp = 0
    global_fp = 0
    global_fn = 0
    list_dq, list_sq, list_pq = [], [], []

    print(f"Évaluation image par image en cours ({len(chemins_images)} images)...")
    
    #  Boucle d'évaluation
    for img_path in tqdm(chemins_images, desc="Inférence et Scores", unit="img"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Recherche du masque (nouveau format PNG ou ancien format TIFF)
        gt_path = None
        for suffix in ['_nuclei_mask.png', '_mask.png', '_masks.png', '_masks.tiff', '_mask.tif', '_masks.tif', '.png']:
            candidate = os.path.join(dossier_lbl, f"{base_name}{suffix}")
            if os.path.exists(candidate):
                gt_path = candidate
                break

        if gt_path is None:
            print(f"  [!] Masque Ground Truth introuvable pour {base_name} - Ignoré")
            continue
        
        # Lecture et sécurisation de l'image
        img = io.imread(img_path)
        if img.shape[-1] > 3:
            img = img[..., :3] # Retire le canal Alpha

        # Lecture INTELLIGENTE du Ground Truth
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        
        if gt_mask is None:
            continue
            
        # Rétrocompatibilité (si c'est un ancien masque RGB 8-bit)
        if gt_mask.ndim > 2:
            gt_mask = gt_mask[:, :, 0]
            
        gt_mask = gt_mask.astype(np.int32)

        # Prédiction de Cellpose
        masks_pred, flows, styles = model.eval(img, diameter=None)
        masks_pred = masks_pred.astype(np.int32)

        # Sécurité de redimensionnement (au cas où)
        if masks_pred.shape != gt_mask.shape:
            masks_pred = cv2.resize(masks_pred.astype(np.float32), 
                                    (gt_mask.shape[1], gt_mask.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST).astype(np.int32)

        pq_metrics, counts, _, _ = get_pq(gt_mask, masks_pred, match_iou=iou_threshold)
        
        dq, sq, pq = pq_metrics
        tp, fp, fn = counts

        # Sauvegarde des scores de l'image
        list_dq.append(dq)
        list_sq.append(sq)
        list_pq.append(pq)
        
        # Cumul pour les scores globaux
        global_tp += tp
        global_fp += fp
        global_fn += fn

    #  Calcul des métriques finales
    avg_dq = np.mean(list_dq) if list_dq else 0
    avg_sq = np.mean(list_sq) if list_sq else 0
    avg_pq = np.mean(list_pq) if list_pq else 0

    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*60)
    print(" RÉSULTATS DE L'ÉVALUATION (Panoptic Quality & COCO) ")
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
    print(" 2 DÉTECTION GLOBALE COCO (Toutes images confondues)")
    print(f"   Vrais Positifs (TP) : {global_tp} cellules correctement trouvées")
    print(f"   Faux Positifs  (FP) : {global_fp} fausses alarmes (bruit/fond)")
    print(f"   Faux Négatifs  (FN) : {global_fn} vraies cellules ratées")
    print(f"   Précision           : {precision*100:05.2f}%")
    print(f"   Rappel (Recall)     : {recall*100:05.2f}%")
    print(f"   F1-Score Global     : {f1_score*100:05.2f}%")
    print("="*60 + "\n")

    return len(list_dq), avg_sq, avg_dq, avg_pq, global_tp, global_fp, global_fn, precision, recall, f1_score



def evaluate_single_image(image_path, gt_path, model_path, iou_threshold=0.5):
    # 1. Choix du device (MPS pour ton Mac, CUDA pour serveur, ou CPU)
    device = deviceChoice()
    
    # 2. Chargement de l'image
    print(f"\n[1/4] Chargement de l'image : {os.path.basename(image_path)}")
    img = io.imread(image_path)
    
    # Correction 4 canaux pour Cellpose
    if img.shape[-1] > 3:
        print(f"     Note : Image à {img.shape[-1]} canaux détectée. Cellpose utilisera les 3 premiers.")

    # 3. Chargement du modèle
    print(f"[2/4] Chargement du modèle...")
    if model_path and os.path.exists(model_path):
        model = models.CellposeModel(gpu=(device.type != 'cpu'), pretrained_model=model_path, device=device)
    else:
        print("     Modèle perso non trouvé, utilisation du modèle 'cyto' par défaut.")
        model = models.CellposeModel(gpu=(device.type != 'cpu'), model_type='cyto', device=device)

    # 4. Segmentation (Inférence)
    print(f"[3/4] Segmentation en cours...")
    masks_pred, flows, styles = model.eval(img, diameter=None, channels=[0,0]) # [0,0] car l'image est déjà chargée
    
    # 5. Évaluation via Panoptic Quality
    print(f"[4/4] Calcul des métriques (IoU >= {iou_threshold})...")
    if os.path.exists(gt_path):
        
        # --- NOUVEAU CHARGEMENT INTELLIGENT ---
        # On force la lecture brute (16-bit) pour ne pas détruire les IDs > 255
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        
        # Sécurité "Ancien Dataset" : Si l'image est quand même en couleurs (3 canaux), 
        # on ne garde que le premier canal. S'il n'y a qu'un canal (Nouveau Dataset),
        # cette ligne est ignorée et tes 640 IDs sont sauvés !
        if gt_mask.ndim > 2:
            gt_mask = gt_mask[:, :, 0]
            
        # Conversion obligatoire en entier 32-bit pour les calculs mathématiques
        gt_mask = gt_mask.astype(np.int32)
        masks_pred = masks_pred.astype(np.int32)
        # --------------------------------------

        print("IDs uniques dans le Ground Truth :", np.unique(gt_mask))
        print("Valeur MAX dans le Ground Truth  :", np.max(gt_mask))

        print("IDs uniques dans la Prédiction   :", np.unique(masks_pred))
        print("Valeur MAX dans la Prédiction    :", np.max(masks_pred))

        print(f"Shape Ground Truth : {gt_mask.shape} | Type : {gt_mask.dtype}")
        print(f"Shape Prediction   : {masks_pred.shape} | Type : {masks_pred.dtype}")
        
        plt.figure(figsize=(10, 10))

        # 1. On affiche la Vérité Terrain (Médecin) en ROUGE
        plt.imshow(gt_mask > 0, cmap='Reds', alpha=0.5)

        # 2. On affiche la Prédiction (Cellpose) en VERT
        plt.imshow(masks_pred > 0, cmap='Greens', alpha=0.5)

        plt.title("Rouge: Ground Truth | Vert: Cellpose | Sombre/Marron: Chevauchement exact")
        plt.axis('off')
        plt.savefig('debug_overlap.png', bbox_inches='tight')
        print("Image de débogage générée : Regarde debug_overlap.png !")

        # Appel à la fonction Panoptic Quality
        pq_metrics, counts, _, _ = get_pq(gt_mask, masks_pred, match_iou=iou_threshold)
        dq, sq, pq = pq_metrics
        tp, fp, fn = counts

        # Affichage des résultats
        print("\n" + "="*50)
        print(f" RÉSULTATS POUR : {os.path.basename(image_path)}")
        print("="*50)
        print(f" SQ (Segmentation Quality) : {sq*100:.2f}% (Précision des contours)")
        print(f" DQ (Detection Quality)    : {dq*100:.2f}% (F1-score / Reconnaissance)")
        print(f" PQ (Panoptic Quality)     : {pq*100:.2f}% (Score global)")
        print("-" * 50)
        print(f" Cellules trouvées (TP)    : {tp}")
        print(f" Fausses alarmes   (FP)    : {fp}")
        print(f" Cellules ratées   (FN)    : {fn}")
        print("="*50)
    else:
        print(" [!] Ground Truth non trouvé, calcul des métriques impossible.")

    # 6. Sauvegarde visuelle
    output_name = "eval_result.png"
    fig = plt.figure(figsize=(15, 5))
    img_rgb = img[:, :, :3] if img.ndim == 3 and img.shape[-1] > 3 else img
    plot.show_segmentation(fig, img_rgb, masks_pred, flows[0])
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"\nVisuel sauvegardé sous : {output_name}")

def main():
    parser = argparse.ArgumentParser(description="Évaluation hybride Cellpose-SAM (CytoDArk & PUMA)")

    # --- 1. Arguments Standard (Toujours requis ou avec défauts) ---
    parser.add_argument("--test_dir", type=str, 
                        default="../cytoDArk_split/20x/256x256/test", 
                        help="Dossier contenant les images et masques")
    parser.add_argument("--model", type=str, 
                        default="baseline", 
                        help="Chemin vers les poids .pt ou 'baseline' pour le modèle standard")
    parser.add_argument("--iou", type=float, 
                        default=0.5, 
                        help="Seuil IoU (par défaut 0.5)")

    # --- 2. Arguments de Reporting (Format PUMA / CSV) ---
    # On les met à None par défaut pour savoir si l'utilisateur veut un CSV ou non
    parser.add_argument("--modele_nom", type=str, default=None, help="Nom d'affichage dans le CSV")
    parser.add_argument("--famille", type=str, default="Cellpose-SAM", help="Famille d'algo (ex: Vista, Cellpose)")
    parser.add_argument("--zoom", type=str, default="Unknown", help="Niveau de zoom ou type de tissu")
    parser.add_argument("--dim", type=str, default="1024", help="Dimension des images")
    parser.add_argument("--csv", type=str, default=None, help="Chemin du fichier CSV de sortie")

    args = parser.parse_args()
    device = deviceChoice()

    # Gestion de la baseline : si l'utilisateur tape --model baseline, on passe None à EvaluateDataset
    model_to_load = None if args.model.lower() == "baseline" else args.model

    # --- 3. Lancement de l'évaluation technique ---
    # EvaluateDataset retourne 10 valeurs d'après ton code source
    results = EvaluateDataset(
        test_dir=args.test_dir, 
        device=device, 
        model_path=model_to_load, 
        iou_threshold=args.iou
    )

    # On dépaquette les résultats
    nb_img, sq, dq, pq, tp, fp, fn, prec, rec, f1 = results

    # --- 4. Logique de Sauvegarde Intelligente ---
    # On n'enregistre dans le CSV que si --csv est présent dans la commande
    if nb_img > 0 and args.csv is not None:
        # Si aucun nom n'est donné, on prend le nom du fichier de modèle
        nom_final = args.modele_nom if args.modele_nom else os.path.basename(args.model)
        
        print(f"\nExportation des métriques vers : {args.csv}")
        save_metrics_to_csv(
            csv_path=args.csv, 
            famille=args.famille, 
            modele=nom_final,
            zoom=args.zoom, 
            dim=args.dim, 
            iou=args.iou, 
            nb_images=nb_img,
            sq=sq, 
            dq=dq, 
            pq=pq, 
            tp=tp, 
            fp=fp, 
            fn=fn, 
            precision=prec, 
            recall=rec, 
            f1=f1
        )
    else:
        print("\nÉvaluation terminée. Aucun export CSV demandé.")

if __name__ == "__main__":
    main()

