import os
import torch
from cellpose import models, io, train

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


def run():

    io.logger_setup()
    device = deviceChoice()

    train_images_paths, train_masks_paths, val_images_paths, val_masks_paths= train_val_split(base_dir="cytoDArk_split")


    print(f"Images trouvées -> Train : {len(train_images_paths)} | Test : {len(val_images_paths)}")


    # Cellpose a besoin que les images soient lues (matrices numpy) avant l'entraînement
    print("Chargement des images en mémoire (cela peut prendre un instant)...")
    train_data = [io.imread(p) for p in train_images_paths]
    train_labels = [io.imread(p) for p in train_masks_paths]

    val_data = [io.imread(p) for p in val_images_paths]
    val_labels = [io.imread(p) for p in val_masks_paths]

    # configs
    model = models.CellposeModel(gpu=True, device=device)

    model_name = "cellpose_dauphin_complet"
    n_epochs = 30   
    learning_rate = 1e-5 
    weight_decay = 0.1
    batch_size = 4   # bon c'est peut être un peu trop, si on a une erreur memoire à l'entrainement faudra baisser ça à 2 voir 1 qui sait     

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


if __name__=="__main__":
    new_model_path, train_losses, test_losses = run()
    print(f"Entraînement terminé ! Modèle sauvegardé ici : {new_model_path}")


