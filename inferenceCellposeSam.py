import torch
from cellpose import models, io, plot, metrics
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import  precision_score, recall_score, f1_score, jaccard_score
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

    #l ance la prédiction 
    print("Segmentation en cours...")
    masks, flows, styles = model.eval(img, diameter=None)

    nombre_cellules = masks.max()
    print(f" Terminé ! {nombre_cellules} cellules trouvées.")

    # sauvegarde le résultat
    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, img, masks, flows[0])
    plt.tight_layout()
    plt.savefig("resultat_serveur.png", dpi=300)
    print("Image sauvegardée sous 'resultat_serveur.png'")
    return  masks, flows ,styles


# méthode utilitaire pour sauvegarder le resultat de la segmentation à la racine
def showResult(img,masks,flows,output_name="resultat_serveur.png"):

    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, img, masks, flows[0])
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"Image sauvegardée sous '{output_name}'")




# --------------- évaluations des perfs ---------------



def evaluate_segmentation(masks, chemin_gt,baseline_black=False):
    # charge le ground truth qui est en .tiff
    print(f"\nChargement du ground truth : {chemin_gt}...")
    gt_image = getImage(chemin_gt)

    #Cellpose renvoie des masques d'instances (1, 2, 3... pour chaque cellule), en gros la premiere cellule a que des 1 en pixel, la 2eme que des 2 etc...
    #Pour comparer  pixel par pixel, on binarise : 
    #Tout ce qui est > 0 est une cellule (True), le reste non 
    pred_binary = masks > 0
    gt_binary = gt_image > 0

    # Ici, on crée une baseline  "Tout en noir" (que des zéros, de la même taille que le Ground truth), c'est la pire prediction possible, rien n'est détecter, on l'utilise à la fin pour vérifier la cohérence des métrics 
    black_baseline = np.zeros_like(gt_binary)
    evaluate_metrics(gt_binary, pred_binary, "Cellpose-SAM")
    evaluate_metrics(gt_binary, black_baseline, "Baseline (Tout Noir)")


#fonction pour calculer les métriques
def evaluate_metrics(y_true, y_pred, name="Modèle"):
    #L'image est en 2d, on l'applatit en une ligne 1d de pixel, on peut traiter ça comme un probleme de classification bianire, 0 fond noir , 1 si constitue 1 cellule 
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()  # le ground truth et la prediction doivent avoir le même format 

    # metrics 
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
        total_tp = np.sum(tp[:, i]) # Toutes les cellules trouvées dans toutes les images
        total_fp = np.sum(fp[:, i]) # Toutes les fausses alarmes
        total_fn = np.sum(fn[:, i]) # Toutes les cellules ratées
        
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

def EvaluateFolder(dossier_test, device, model_path=None):

    print(f"\n=== Lancement de l'évaluation sur le dossier : {dossier_test} ===")
    
    chemins_images = sorted([os.path.join(dossier_test, f) for f in os.listdir(dossier_test) if f.endswith('.png')])
    chemins_labels = [p.replace('.png', '_masks.tiff') for p in chemins_images]
    
    # Chargement en mémoire
    test_data = [getImage(p) for p in chemins_images]
    test_labels = [getImage(p) for p in chemins_labels]
    
    # modeele
    if model_path:
        model = models.CellposeModel(gpu=True, pretrained_model=model_path, device=device)
    else:
        model = models.CellposeModel(gpu=True, device=device)
        
    print(f"Prédiction en lot sur {len(test_data)} images en cours...")
    masks_pred, flows, styles = model.eval(test_data, diameter=None)
    
    # les metrics de cellpose 
    seuils_iou = [0.5, 0.75, 0.9]
    ap, tp, fp, fn = metrics.average_precision(test_labels, masks_pred, threshold=seuils_iou)
    
    #affichage des scores
    print_coco_metrics(tp, fp, fn, seuils_iou)


def main(): 
    device = deviceChoice() # on mets cette méthode à l'extèrieur pour éviter de perdre du temps à charger à chaque image le device 
    
    modele_perso = "models/cellpose_dauphin_complet" 

    
    
    print("\n--- TEST VISUEL SUR UNE IMAGE ---")
    # Prends une image au hasard dans ton dossier de TEST (le coffre-fort)
    chemin_image = "cytoDArk_split/20x/256x256/test/ID5_Aud_Cortex_Tursiops_1.png" 
    chemin_gt="cytoDArk_split/20x/256x256/test/ID5_Aud_Cortex_Tursiops_1_masks.tiff"
    
    
    masks, flows, styles= SegmentOneImage(chemin_image, device=device, model_path=modele_perso)

    img= getImage(chemin_image)
    showResult(img,masks,flows)
    
    # On sauvegarde le résultat
    showResult(img, masks, flows, output_name="resultat_dauphin.png")


    print("\n--- ÉVALUATION chiffré ---")
    evaluate_segmentation(masks,chemin_gt=chemin_gt,baseline_black=True )

    """
    #Si tu veux lancer le test sur un dossier test : 

    dossier_test_20_256 ="cytoDArk_split/20x/256x256/test"
    EvaluateFolder(dossier_test=dossier_test_20_256, device=device,model_path=modele_perso)
    """

if __name__ == "__main__":
    main()

