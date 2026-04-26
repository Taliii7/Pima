"""
Convertit le dataset PUMA pour le rendre compatible avec VISTA2D :
  - Images RGBA (4 canaux) → RGB (3 canaux), conservées en .tif
  - Masques PNG 16-bit (mode I;16) → TIFF 32-bit entier, renommés en .tiff

Les fichiers convertis sont écrits dans un nouveau dossier (puma_data_converted/)
pour ne pas modifier l'original.

Usage :
    python convert_puma_dataset.py --src ../puma_data --dst ../puma_data_converted
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_image(src_path: Path, dst_path: Path):
    """RGBA/RGB .tif → RGB .tif"""
    img = Image.open(src_path)
    arr = np.array(img)

    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]  # drop alpha channel

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(dst_path)


def convert_mask(src_path: Path, dst_path: Path):
    """PNG 16-bit → TIFF 32-bit entier (instance labels)"""
    img = Image.open(src_path)
    arr = np.array(img).astype(np.int32)  # 16-bit → 32-bit signé

    # Nouveau nom : remplace _nuclei_mask.png → _nuclei_mask.tiff
    dst_path = dst_path.with_suffix(".tiff")
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarder en TIFF 32-bit via PIL (mode I = 32-bit signed int)
    Image.fromarray(arr, mode="I").save(dst_path)

    return dst_path  # retourner le vrai chemin (extension changée)


def process_split(src_split: Path, dst_split: Path):
    images_src = src_split / "images"
    masks_src  = src_split / "masks"
    images_dst = dst_split / "images"
    masks_dst  = dst_split / "masks"

    # --- Images ---
    img_files = sorted(images_src.iterdir())
    print(f"  Images ({len(img_files)}) ...")
    for src in tqdm(img_files, leave=False):
        if src.suffix.lower() not in (".tif", ".tiff", ".png"):
            continue
        convert_image(src, images_dst / src.name)

    # --- Masques ---
    mask_files = sorted(masks_src.iterdir())
    print(f"  Masques ({len(mask_files)}) ...")
    for src in tqdm(mask_files, leave=False):
        if src.suffix.lower() not in (".png", ".tif", ".tiff"):
            continue
        convert_mask(src, masks_dst / src.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="../puma_data",
                        help="Dossier source (défaut: ../puma_data)")
    parser.add_argument("--dst", type=str, default="../puma_data_converted",
                        help="Dossier de destination (défaut: ../puma_data_converted)")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    for split in ["train", "val", "test"]:
        src_split = src_root / split
        if not src_split.exists():
            print(f"[{split}] absent, ignoré.")
            continue
        print(f"\n[{split}]")
        process_split(src_split, dst_root / split)

    print(f"\n✅ Conversion terminée → {dst_root}")
    print("Pense à mettre à jour basedir dans ton YAML :")
    print(f"  basedir: {dst_root}")
    print("Et à regénérer le datalist (les masques sont maintenant en .tiff) :")
    print("  python generate_puma_datalist.py --data_root puma_data_converted \\")
    print("                                    --output datalists/puma_datalist.json")


if __name__ == "__main__":
    main()