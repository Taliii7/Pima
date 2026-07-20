import sys
import os
import argparse
import numpy as np
from skimage.io import imread
from tqdm import tqdm

# Imports spécifiques à StarDist
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# Ajout du dossier 'common' pour récupérer tes fonctions utilitaires
# Remplace les lignes de dossier_common par :
# Ajout du dossier 'common' pour récupérer tes fonctions utilitaires
import sys
import os

# 1. Utilisation de abspath pour résoudre proprement les ".." (et les problèmes liés au NFS)
dossier_common = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common'))

if not os.path.exists(dossier_common):
    print(f"ATTENTION : Le dossier {dossier_common} n'existe pas physiquement.")
    
if dossier_common not in sys.path:
    sys.path.append(dossier_common)

try:
    # Import de tes fonctions d'évaluation maison
    from evaluate_models import get_pq
    from utils import save_metrics_to_csv
except ImportError as e:
    # 2. On affiche l'erreur EXACTE (très important pour le debug)
    print(f"\n[!] CRASH D'IMPORTATION : {e}")
    print(f"[!] Dossier cherché : {dossier_common}")
    sys.exit(1)
def EvaluateDataset(test_dir, model_name, iou_threshold=0.5):
    """
    Évalue le modèle StarDist sur tout un dossier.
    Gère la normalisation requise par csbdeep et retourne les métriques globales.
    """
    print(f"\n=== Lancement de l'évaluation StarDist sur le dossier : {test_dir} ===")
    
    if not os.path.exists(test_dir):
        print(f"Erreur : Le dossier {test_dir} n'existe pas.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    #  Gestion intelligente des chemins (Support PUMA / CytoDArk)
    img_dir = os.path.join(test_dir, "images")
    mask_dir = os.path.join(test_dir, "masks")
    
    #au cas où 
    if not os.path.exists(img_dir):
        img_dir = test_dir
        mask_dir = test_dir
        
    images_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) 
                    if f.endswith(('.tif', '.png', '.tiff')) 
                    and not f.endswith('_masks.tiff') 
                    and not f.endswith('_nuclei_mask.png')]
    
    if len(images_paths) == 0:
        print("Erreur : Aucune image trouvée pour l'évaluation.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Chargement du Modèle StarDist
    try:
        print(f"Chargement du modèle StarDist : {model_name}")
        # On peut charger '2D_versatile_fluo' ou un modèle fine-tuné local
        local_path = os.path.join('models', model_name)
        if os.path.exists(local_path):
        # Cherche un sous-dossier (cas où StarDist a sauvegardé dans un sous-dossier)
            subdirs = [d for d in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, d))]
            if subdirs:
                model = StarDist2D(None, subdirs[0], basedir=local_path)
            else:
                model = StarDist2D(None, model_name, basedir='models')
        else:
            model = StarDist2D.from_pretrained(model_name)
    except Exception as e:
        print(f"Erreur de chargement du modèle {model_name}. Erreur: {e}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # vaaariables pour accumuler les métriques de toutes les images
    # teeest
    # all_sd, all_pq= 0, 0

    all_sq, all_dq, all_pq = [], [], []
    global_tp, global_fp, global_fn = 0, 0, 0
    
    print(f"Évaluation de {len(images_paths)} images en cours...")
    for img_path in tqdm(images_paths):
        #print("iiiiiii",img_path)
        # Construction du nom du masque (PUMA ou CytoDArk)
        base_name = os.path.basename(img_path)
        #print("base name :" , base_name)
        name_no_ext = os.path.splitext(base_name)[0]
        #print("base name no ext :",name_no_ext)
        mask_path = os.path.join(mask_dir, f"{name_no_ext}_nuclei_mask.tiff") # PUMA (converted)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, f"{name_no_ext}_masks.tiff") # CytoDArk
        
        if not os.path.exists(mask_path):
            print("NOO",mask_path)
            continue
            
        # lecture de l'image et du masks ground thuth
        img = imread(img_path)
        mask_true = imread(mask_path)

        if img.ndim == 3:
            if img.shape[-1] == 4:
                img = img[..., :3]  # RGBA = > RGB, garder les 3 canaux
            # Si déjà RGB (3 canaux) => ne rien faire
        elif img.ndim > 3:
            img = np.squeeze(img)
                # -----------------------
        # Inférence StarDist
        #normalisation entre les percentiles 1 et 99.8,  on est obligé de faire ça car les chercheurs qui ont bosser sur ce model l'ont entrainé pour envoyer en sortie des valeurs entre 0 et 1, si on fait pas cette normalisation, on aura un masque absurde
        img_norm = normalize(img, 1, 99.8, axis=(0,1))
        labels_pred, _ = model.predict_instances(img_norm)
        #print("img norm",img_norm)
   

        pq_metrics, counts, _, _ = get_pq(mask_true, labels_pred, iou_threshold)
    
        dq, sq, pq = pq_metrics
        tp, fp, fn = counts

        # Sauvegarde des scores de l'image
        all_dq.append(dq)
        all_sq.append(sq)
        all_pq.append(pq)
        
        # Cumul pour les scores globaux
        global_tp += tp
        global_fp += fp
        global_fn += fn

    #  Calcul des métriques finales
    avg_dq = np.mean(all_dq) if all_dq else 0
    avg_sq = np.mean(all_sq) if all_sq else 0
    avg_pq = np.mean(all_pq) if all_pq else 0

    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*60)
    print(" RÉSULTATS DE L'ÉVALUATION (Panoptic Quality & COCO) ")
    print("="*60)
    print(f" Modèle évalué     : {model_name}")
    print(f" Seuil de rigueur  : IoU >= {iou_threshold}")
    print(f" Images traitées   : {len(all_dq)}")
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

    #print('retourne ', len(all_dq), avg_sq, avg_dq, avg_pq, global_tp, global_fp, global_fn, precision, recall, f1_score)
    return len(all_dq), avg_sq, avg_dq, avg_pq, global_tp, global_fp, global_fn, precision, recall, f1_score



def main():
    parser = argparse.ArgumentParser(description="Évaluation StarDist")
    parser.add_argument("--test_dir", type=str, required=True, help="Dossier contenant les images de test")
    parser.add_argument("--model", type=str, default="2D_versatile_he", help="Nom du modèle StarDist")
    parser.add_argument("--iou", type=float, default=0.5)
    
    # Arguments pour le CSV (communs aux autres scripts)
    parser.add_argument("--famille", type=str, default="StarDist")
    parser.add_argument("--modele_nom", type=str, required=True, help="Nom lisible pour le CSV (ex: StarDist_Baseline)")
    parser.add_argument("--zoom", type=str, required=True)
    parser.add_argument("--dim", type=str, required=True)
    parser.add_argument("--csv", type=str, default="../resultats_comparaison/results_test.csv")
    args = parser.parse_args()
    
    results = EvaluateDataset(
        test_dir=args.test_dir, 
        model_name=args.model, 
        iou_threshold=args.iou
    )

    nb_img, sq, dq, pq, tp, fp, fn, prec, rec, f1 = results

    # Sauvegarde dans le CSV
    if nb_img > 0 and args.csv is not None:
        print(f"\nExportation des métriques vers : {args.csv}")
        save_metrics_to_csv(
            csv_path=args.csv, 
            famille=args.famille, 
            modele=args.modele_nom,
            zoom=args.zoom, 
            dim=args.dim, 
            iou=args.iou, 
            nb_images=nb_img,
            sq=sq, dq=dq, pq=pq, 
            tp=tp, fp=fp, fn=fn, 
            precision=prec, recall=rec, f1=f1
        )
        print("="*60)
        print(f" RÉSULTATS {args.modele_nom} : F1={f1*100}% | PQ={pq*100}% | TP={tp} | FN={fn} | FP={fp}")
        print("="*60)
    else:
        print(" [!] Évaluation échouée : aucune donnée à sauvegarder.")
        #print("nb images",nb_img)
        #print("csv",args.csv)

if __name__ == "__main__":
    main()