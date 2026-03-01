import torch
from cellpose import models, io, plot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import  precision_score, recall_score, f1_score, jaccard_score



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


def SegmentOneImage(chemin_image,device):

    # charge l'image
    print(f"Chargement de l'image : {chemin_image}...")
    img = getImage(chemin_image)

    print(f"Image de dimension : {img.shape}. L'image a en dimension de channels  {img.shape[-1]} ")

    # charge le modèle Cellpose-Sam 
    print("Chargement du modèle Cellpose-Sam")
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
def showResult(img,masks,flows):

    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, img, masks, flows[0])
    plt.tight_layout()
    plt.savefig("resultat_serveur.png", dpi=300)
    print("Image sauvegardée sous 'resultat_serveur.png'")




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




def main(): 

    # Change ces chemins en fonction de là où se trouvent ces fichiers, que ce soit dans ton ordi ou dans ta session virtuelle dans les serveurs de la fac
    chemin_image = "cytoDArk0/20x/256x256/image/ID1_Aud_Cortex_Tursiops_1.png" 
    chemin_gt = "cytoDArk0/20x/256x256/label/ID1_Aud_Cortex_Tursiops_1.tiff" # faire attention que l'id et le chiffre à lafin correspondent bien à l'id et le chiffre de fin de l'image originel

    device = deviceChoice() # on mets cette méthode à l'extèrieur pour éviter de perdre du temps à charger à chaque image le device 
    masks, flows ,styles = SegmentOneImage(chemin_image,device=device)

    img= getImage(chemin_image)
    showResult(img,masks,flows)

    evaluate_segmentation(masks,chemin_gt=chemin_gt,baseline_black=True) # je mets à True, juste pour le test, mais on pourra enlever plus tard




if __name__ == "__main__":
    main()
