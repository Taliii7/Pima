import sys
import argparse
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_dir, '..', 'common')

if common_path not in sys.path:
    sys.path.append(common_path)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellpose import models, io, train
import matplotlib.pyplot as plt
from cyrk0_split import train_val_split

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

def plot(args, train_losses, val_losses):
    # --- SAUVEGARDE DE LA COURBE DE LOSS ---
    print("Génération de la courbe d'apprentissage...")
    plt.figure(figsize=(10, 6))
    
    # On trace les deux listes (x = numéro de l'époque, y = valeur de l'erreur)
    plt.plot(train_losses, label='Train Loss (Apprentissage)', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss (Examen blanc)', color='orange', linewidth=2)
    
    plt.title(f"Courbe d'entraînement : {args.output_name}")
    plt.xlabel('Époques')
    plt.ylabel('Loss (Erreur)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # On sauvegarde l'image au même endroit que le script
    nom_image = f"Loss_curve_{args.output_name}.png"
    plt.savefig(nom_image, dpi=300, bbox_inches='tight')
    print(f"Super ! Courbe sauvegardée sous : {nom_image}")

def run(args):

    io.logger_setup()
    device = deviceChoice()

    train_images_paths, train_masks_paths = [], []
    val_images_paths, val_masks_paths = [], []

    # on gère les 2 datasets,
    if args.dataset_type == "cyto":
        print(">> Mode CytoDArk activé.")
        train_images_paths, train_masks_paths, val_images_paths, val_masks_paths = train_val_split(
            base_dir=args.base_dir, zoom=args.zoom, dim=args.dim
        )
        
    elif args.dataset_type == "puma":
        print(f">> Mode PUMA activé. Recherche dans {args.base_dir}...")
        
        # fonction utilitaire pour récupérer les paires image/masque de PUMA (fait par un llm)
        def get_puma_paths(split_name):
            img_dir = os.path.join(args.base_dir, split_name, "images")
            mask_dir = os.path.join(args.base_dir, split_name, "masks")
            imgs, masks = [], []
            
            if os.path.exists(img_dir) and os.path.exists(mask_dir):
                for img_name in sorted(os.listdir(img_dir)):
                    if img_name.endswith('.tif') or img_name.endswith('.png'):
                        # Construction du nom du masque correspondant
                        base_name = os.path.splitext(img_name)[0]
                        mask_name = f"{base_name}_nuclei_mask.png"
                        mask_path = os.path.join(mask_dir, mask_name)
                        
                        if os.path.exists(mask_path):
                            imgs.append(os.path.join(img_dir, img_name))
                            masks.append(mask_path)
            return imgs, masks

        # on remplit  les listes PUMA
        train_images_paths, train_masks_paths = get_puma_paths("train")
        val_images_paths, val_masks_paths = get_puma_paths("val")
        
    else:
        print(f"Erreur : Le type de dataset '{args.dataset_type}' n'est pas reconnu.")
        sys.exit(1)
    # -----------------------------

    print(f"Images trouvées -> Train : {len(train_images_paths)} | Test : {len(val_images_paths)}")

    if len(train_images_paths) == 0:
        print("Erreur : Aucune image d'entraînement trouvée. Vérifie tes chemins !")
        sys.exit(1)

    # Cellpose a besoin que les images soient lues (matrices numpy) avant l'entraînement
    print("Chargement des images en mémoire (cela peut prendre un instant)...")
    train_data = [io.imread(p) for p in train_images_paths]
    train_labels = [io.imread(p) for p in train_masks_paths]

    val_data = [io.imread(p) for p in val_images_paths]
    val_labels = [io.imread(p) for p in val_masks_paths]

    # configs
    model = models.CellposeModel(gpu=True, device=device)

    model_name = args.output_name
    n_epochs = args.epochs
    learning_rate = args.lr
    weight_decay = args.wd
    batch_size = args.batch_size  
  
    print("Début de l'entraînement...")
    new_model_path, train_losses, val_losses = train.train_seg(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        test_data=val_data,
        test_labels=val_labels,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        nimg_per_epoch=max(2, len(train_data)),
        model_name=model_name,
        save_each=True,  # Sauvegarde des modèles intermédiaires
        save_every=5     # Sauvegarde à CHAQUE époque
    )
    return new_model_path, train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(description="Script pour entrainer Cellpose")

    # --- nouveau paraa ici, pour faire en sorte que ce fichier soit modulaire pour le dataset cyto et puma, bon pour les nouveaux datasets, ça risque d'être chiant mais bon on y est pas encore
    parser.add_argument("--dataset_type", required=True,type=str, choices=["cyto", "puma"], help="Choix du dataset: 'cyto' ou 'puma'")

    # Entrée / Sortie
    parser.add_argument("--base_dir", type=str, default="cytoDArk_split", help="Le dossier racine du dataset")
    parser.add_argument("--output_name", type=str, default="cellpose_dauphin_complet", help="Le nom de fichier de sortie")

    # Sur quoi on veut entrainer le modele (utilisé que pour Cyto)
    parser.add_argument("--zoom", type=int, default=0, help="Le zoom (0=all, 20=20x, 40=40x)")
    parser.add_argument("--dim", type=int, default=0, help="La dimension (0=all, 1=256, blablabl)")
    
    # Paramètres du modèle 
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.1, help="weight_decay")
    parser.add_argument("--batch_size", type=int, default=4, help="Le nombre de batchs")
    parser.add_argument("--epochs", type=int, default=30, help="Le nombre d'époques")

    args = parser.parse_args()

    new_model_path, train_losses, val_losses = run(args)
    #detection du meilleur model basé sur la meilleur loss de validation, afin d'éviter l'overfitting. Contrairement à Vista, ça n'est pas proposé par défaut
    print("les valeurs de loss : ", val_losses)    
    best_epoch = np.argmin(val_losses)
    best_loss = val_losses[best_epoch]
    print(f"\n=======================================================")
    print(f" RÉSULTAT ANTI-OVERFITTING :")
    print(f"Le meilleur modèle absolu a été vu à l'époque : {best_epoch}")
    print(f"Avec une validation loss minimale de : {best_loss:.4f}")
    print(f"Le fichier correspondant s'appelle normalement : {args.output_name}_epoch_{best_epoch}")
    print(f"=======================================================\n")
    # ---------------------------------------------

    print(f"Entraînement terminé ! Dernier modèle sauvegardé ici : {new_model_path}")
    print(f"Entraînement terminé ! Modèle sauvegardé ici : {new_model_path}")

    plot(args, train_losses=train_losses, val_losses=val_losses)

if __name__=="__main__":
    main()
