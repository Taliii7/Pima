"""

Structure attendue du dataset :
    puma_data/
    ├── train/
    │   ├── images/   (*.tif)
    │   └── masks/    (*_nuclei_mask.png)
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/


"""

import argparse
import json
import os
from pathlib import Path


def get_pairs(split_dir: Path) -> list[dict]:
    """
    Associe chaque image à son masque correspondant.
    Convention : l'image s'appelle <stem>.tif
                 le masque s'appelle <stem>_nuclei_mask.png
    """
    images_dir = split_dir / "images"
    masks_dir  = split_dir / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {masks_dir}")

    pairs = []
    missing_masks = []

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in (".tif", ".tiff", ".png"):
            continue

        expected_mask = masks_dir / f"{img_path.stem}_nuclei_mask.tiff"

        if not expected_mask.exists():
            missing_masks.append(img_path.name)
            continue

        pairs.append({
            "image": f"{split_dir.name}/images/{img_path.name}",
            "label": f"{split_dir.name}/masks/{expected_mask.name}",
        })

    if missing_masks:
        print(f"    {len(missing_masks)} image(s) sans masque ignorée(s) :")
        for m in missing_masks:
            print(f"       - {m}")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Génère puma_datalist.json")
    parser.add_argument(
        "--data_root",
        type=str,
        default="puma_data",
        help="Chemin vers le dossier racine du dataset PUMA (défaut: puma_data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datalists/puma_datalist.json",
        help="Chemin de sortie du fichier JSON (défaut: datalists/puma_datalist.json)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root introuvable : {data_root}")

    splits = {
        "training":   "train",
        "validation": "val",
        "testing":    "test",
    }

    datalist = {}
    for json_key, folder_name in splits.items():
        split_dir = data_root / folder_name
        print(f"[{json_key}] lecture de {split_dir} ...")

        if not split_dir.exists():
            print(f"    Dossier absent, split vide.")
            datalist[json_key] = []
            continue

        pairs = get_pairs(split_dir)
        datalist[json_key] = pairs
        print(f"   {len(pairs)} paires trouvées")

    # Créer le dossier de sortie si nécessaire
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(datalist, f, indent=2)

    print(f"\n Fichier généré : {output_path}")
    print(f"   training:   {len(datalist['training'])} items")
    print(f"   validation: {len(datalist['validation'])} items")
    print(f"   testing:    {len(datalist['testing'])} items")


if __name__ == "__main__":
    main()