import sys
import argparse
import os
import torch

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


    # --- NOUVEAU : SAUVEGARDE DE LA COURBE DE LOSS ---
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

    train_images_paths, train_masks_paths, val_images_paths, val_masks_paths= train_val_split(base_dir=args.base_dir,zoom=args.zoom, dim = args.dim)


    print(f"Images trouvées -> Train : {len(train_images_paths)} | Test : {len(val_images_paths)}")


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

    print("Début de l'entraînement global...")
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
        model_name=model_name
    )
    return new_model_path, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description="Script pour diviser le dataset cytoDArk")

    #entrée / sortie
    parser.add_argument("--base_dir", type=str, default="cytoDArk_split", help="Le dossier racine")
    parser.add_argument("--output_name", type=str, default="cellpose_dauphin_complet", help="Le nom de fichier de sortie")

    # Sur quoi on veut entrainer le modele 
    parser.add_argument("--zoom", type=int, default=0, help="Le zoom (0=all, 20=20x, 40=40x)")
    parser.add_argument("--dim", type=int, default=0, help="La dimension (0=all, 1=256, blablabl)")
    #les paramètres du modèle 
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.1, help="weight_decay")
    parser.add_argument("--batch_size", type=int, default=4, help="LLe nombre de batchs")
    parser.add_argument("--epochs", type=int, default=30, help="Le nombre d'époques")


    args = parser.parse_args()

    new_model_path, train_losses, val_losses = run(args)
    print(f"Entraînement terminé ! Modèle sauvegardé ici : {new_model_path}")

    plot(args,train_losses=train_losses,val_losses=val_losses)



if __name__=="__main__":
    main()
   


