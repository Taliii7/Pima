"""
Script d'entraînement StarDist (fine-tuning depuis 2D_versatile_he)
Compatible avec les datasets CytoDArk et PUMA.

Usage PUMA :
    CUDA_VISIBLE_DEVICES=3 nohup python3 train_StarDist.py \
        --dataset_type puma \
        --base_dir ../puma_data_converted \
        --output_name stardist_puma \
        --epochs 70 \
        > journal_stardist_puma.log 2>&1 &

Usage CytoDArk :
    CUDA_VISIBLE_DEVICES=3 nohup python3 train_StarDist.py \
        --dataset_type cyto \
        --base_dir ../cytoDArk_split \
        --output_name stardist_cytodark \
        --zoom 20 \
        --dim 256 \
        --epochs 50 \
        > journal_stardist_cytodark.log 2>&1 &
"""

import os
import sys
import argparse
import shutil
import numpy as np
from skimage.io import imread
from csbdeep.utils import normalize
from stardist.models import StarDist2D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_puma_paths(base_dir, split_name):
    """Récupère les paires image/masque pour le dataset PUMA."""
    img_dir  = os.path.join(base_dir, split_name, "images")
    mask_dir = os.path.join(base_dir, split_name, "masks")
    imgs, masks = [], []

    if not os.path.exists(img_dir):
        print(f"  [!] Dossier introuvable : {img_dir}")
        return imgs, masks

    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.endswith(('.tif', '.tiff', '.png')):
            continue
        base_name = os.path.splitext(img_name)[0]
        for mask_suffix in ['_nuclei_mask.tiff', '_nuclei_mask.png']:
            mask_path = os.path.join(mask_dir, f"{base_name}{mask_suffix}")
            if os.path.exists(mask_path):
                imgs.append(os.path.join(img_dir, img_name))
                masks.append(mask_path)
                break

    return imgs, masks


def get_cyto_paths(base_dir, zoom, dim):
    """Récupère les paires image/masque pour le dataset CytoDArk."""
    try:
        from cyrk0_split import train_val_split
        train_imgs, train_masks, val_imgs, val_masks = train_val_split(
            base_dir=base_dir, zoom=zoom, dim=dim
        )
        return train_imgs, train_masks, val_imgs, val_masks
    except ImportError:
        print("Erreur : impossible d'importer cyrk0_split.")
        sys.exit(1)


def load_images(img_paths, mask_paths):
    """Charge et normalise les images + masques."""
    X, Y = [], []
    for img_path, mask_path in zip(img_paths, mask_paths):
        img  = imread(img_path)
        mask = imread(mask_path).astype(np.int32)

        if img.ndim == 3:
            if img.shape[-1] == 4:
                img = img[..., :3]       # RGBA → RGB
            elif img.shape[0] in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))  # CHW → HWC
        elif img.ndim == 2:
            img = np.stack([img]*3, axis=-1)  # Grayscale → RGB

        img_norm = normalize(img, 1, 99.8, axis=(0, 1))
        X.append(img_norm)
        Y.append(mask)

    return X, Y


def run(args):

    # ==========================================
    # 1. CHARGEMENT DES DONNÉES
    # ==========================================
    print(f">> Mode {args.dataset_type.upper()} activé. Chargement des données...")

    if args.dataset_type == "puma":
        train_img_paths, train_mask_paths = get_puma_paths(args.base_dir, "train")
        val_img_paths,   val_mask_paths   = get_puma_paths(args.base_dir, "val")
    else:
        train_img_paths, train_mask_paths, val_img_paths, val_mask_paths = get_cyto_paths(
            args.base_dir, args.zoom, args.dim
        )

    print(f"   Train : {len(train_img_paths)} images | Val : {len(val_img_paths)} images")

    if len(train_img_paths) == 0:
        print("Erreur : aucune image d'entraînement trouvée !")
        sys.exit(1)

    print("Chargement et normalisation des images en mémoire...")
    X_train, Y_train = load_images(train_img_paths, train_mask_paths)
    X_val,   Y_val   = load_images(val_img_paths,   val_mask_paths)
    print("Chargement terminé")

    # ==========================================
    # 2. CHARGEMENT DU MODÈLE PRÉ-ENTRAÎNÉ
    #    FIX : on crée un nouveau modèle avec la
    #    config du pré-entraîné, puis on charge
    #    ses poids — StarDist sauvegarde alors
    #    dans output_dir/output_name/ dès le départ
    # ==========================================
    print(f"Chargement du modèle pré-entraîné : {args.pretrained_model}")

    # Étape 1 : télécharger/charger le modèle source pour récupérer sa config
    model_source = StarDist2D.from_pretrained(args.pretrained_model)

    # Étape 2 : chemin des poids pré-entraînés dans le cache keras
    pretrained_weights = os.path.join(
        os.path.expanduser('~'), '.keras', 'models', 'StarDist2D',
        args.pretrained_model, 'weights_best.h5'
    )
    if not os.path.exists(pretrained_weights):
        print(f"Erreur : poids pré-entraînés introuvables : {pretrained_weights}")
        sys.exit(1)

    # Étape 3 : créer le nouveau modèle avec le bon nom et dossier de sortie
    save_dir = os.path.join(args.output_dir, args.output_name)
    os.makedirs(save_dir, exist_ok=True)
    model = StarDist2D(model_source.config, name=args.output_name, basedir=args.output_dir)

    # Étape 4 : charger les poids pré-entraînés dans ce nouveau modèle
    model.keras_model.load_weights(pretrained_weights)
    print(f"Poids pré-entraînés chargés depuis : {pretrained_weights}")
    print(f"Le modèle sera sauvegardé dans : {save_dir}/")

    # ==========================================
    # 3. ENTRAÎNEMENT
    # ==========================================
    print(f"Début du fine-tuning : {args.epochs} epochs, {args.steps_per_epoch} steps/epoch")
    model.train(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
    )

    # ==========================================
    # 4. OPTIMISATION DES SEUILS
    # ==========================================
    print("Optimisation des seuils sur le set de validation...")
    model.optimize_thresholds(X_val, Y_val)

    print(f"\n✅ Entraînement terminé ! Modèle sauvegardé dans : {save_dir}/")
    print(f"   → weights_best.h5  (meilleur modèle sur val_loss)")
    print(f"   → weights_last.h5  (dernier epoch)")
    print(f"   → thresholds.json  (seuils optimisés)")


def main():
    parser = argparse.ArgumentParser(description="Entraînement StarDist (fine-tuning)")

    parser.add_argument("--dataset_type", required=True, choices=["puma", "cyto"],
                        help="Type de dataset : 'puma' ou 'cyto'")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Dossier racine du dataset")
    parser.add_argument("--output_name", type=str, default="stardist_finetuned",
                        help="Nom du modèle de sortie")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Dossier de sauvegarde du modèle (défaut: models/)")

    parser.add_argument("--zoom", type=int, default=0,
                        help="Zoom CytoDArk (0=all, 20=20x, 40=40x)")
    parser.add_argument("--dim", type=int, default=0,
                        help="Dimension CytoDArk (0=all, 256, 512...)")

    parser.add_argument("--epochs", type=int, default=70,
                        help="Nombre d'epochs (défaut: 70)")
    parser.add_argument("--steps_per_epoch", type=int, default=100,
                        help="Steps par epoch (défaut: 100)")
    parser.add_argument("--pretrained_model", type=str, default="2D_versatile_he",
                        help="Modèle pré-entraîné de départ (défaut: 2D_versatile_he)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()