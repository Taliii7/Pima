import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.cyrk0_split import train_val_split


# ==========================================
# 2. CONVERTISSEUR POUR MONAI
# ==========================================
def generate_monai_json(base_dir, zoom, dim, output_json):
    if not os.path.exists(base_dir):
        print("bizarre")
    else:
        print("cool",base_dir)

    # Appel de ta fonction
    train_imgs, train_masks, val_imgs, val_masks = train_val_split(base_dir, zoom, dim)
    print(train_imgs, train_masks, val_imgs, val_masks)
    datalist = {"training": [], "validation": [], "testing": []}

    # Formatage pour le Training
    for img, mask in zip(train_imgs, train_masks):
        # os.path.relpath retire "../cytoDArk_split/" du chemin pour plaire à MONAI
        rel_img = os.path.relpath(img, base_dir).replace("\\", "/")
        rel_mask = os.path.relpath(mask, base_dir).replace("\\", "/")
        datalist["training"].append({"image": rel_img, "label": rel_mask})

    # Formatage pour la Validation
    for img, mask in zip(val_imgs, val_masks):
        rel_img = os.path.relpath(img, base_dir).replace("\\", "/")
        rel_mask = os.path.relpath(mask, base_dir).replace("\\", "/")
        datalist["validation"].append({"image": rel_img, "label": rel_mask})

    # Sauvegarde
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(datalist, f, indent=4)
        
    print(f"\n🎉 Fichier généré avec succès : {output_json}")
    print(f"📊 Training   : {len(datalist['training'])} paires")
    print(f"📊 Validation : {len(datalist['validation'])} paires")


# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    # ---> PARAMÈTRES À CHANGER ICI <---
    BASE_DIR = "../cytoDArk_split"  
    ZOOM = 0  # 0 pour toutes les magnifications, 20, ou 40
    DIM = 0   # 0 pour toutes les tailles, 256, 512, 1024, 2048
    
    OUTPUT_JSON = "datalists/cytodark_datalist.json"
    
    generate_monai_json(BASE_DIR, ZOOM, DIM, OUTPUT_JSON)