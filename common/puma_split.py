import os
import glob
import json
import numpy as np
import cv2
import argparse
import random
import shutil
import math

#Pour avoir train/val/test il faut lancer cette fonctin en premier :


def process_instance_segmentation(input_dir, train_dir, vis_dir, width, height):
    if not os.path.isdir(input_dir):
        print(f"Erreur : Le dossier '{input_dir}' n'existe pas.")
        return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    geojson_files = glob.glob(os.path.join(input_dir, '*.geojson'))
    
    if not geojson_files:
        print(f"Aucun fichier .geojson trouvé dans '{input_dir}'.")
        return
        
    print(f"{len(geojson_files)} fichiers trouvés. Début de la génération des instances...\n" + "-"*40)

    for file_path in geojson_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Erreur avec {file_path} : {e}")
            continue

        #impro de ma part, on test avec un masque 16-bit pour supporter > 255 instances, il y a beaucoup plus d'instance dans ce dataset que dans cyto
        instance_mask = np.zeros((height, width), dtype=np.uint16)
        # en 8 bits, on s'en fout ici c'est juste pour le visuel, pas besoin d'avoir 1 chiffre par instance comme avec le masque 
        visual_mask = np.zeros((height, width, 3), dtype=np.uint8)

        features = data.get('features', [])
        
        # o parcourt les annotations, "instance_id" commence à 1 (le 0 c'est le fond)
        for instance_id, feature in enumerate(features, start=1):
            geom_type = feature.get('geometry', {}).get('type')
            if geom_type != 'Polygon':
                continue

            coordinates = feature.get('geometry', {}).get('coordinates', [])
            
            #pour le dessin
            exterior_ring = np.array(coordinates[0], np.int32)
            
            # On attribue l'ID unique à ce noyau
            cv2.fillPoly(instance_mask, [exterior_ring], instance_id)
            
            # on génère une couleur aléatoire basée sur l'ID pour le visuel
            np.random.seed(instance_id)
            random_color = np.random.randint(50, 255, size=3).tolist()
            cv2.fillPoly(visual_mask, [exterior_ring], random_color)

            # Gestion des trous potentiels
            if len(coordinates) > 1:
                for hole in coordinates[1:]:
                    hole_ring = np.array(hole, np.int32)
                    cv2.fillPoly(instance_mask, [hole_ring], 0)
                    cv2.fillPoly(visual_mask, [hole_ring], (0, 0, 0))

        #save
        train_out_path = os.path.join(train_dir, f"{base_name}_mask.png")
        vis_out_path = os.path.join(vis_dir, f"{base_name}_vis.png")
        
        # cv2.imwrite gere automatiquement le format 16bit si le tableau numpy est en uint16
        cv2.imwrite(train_out_path, instance_mask)
        cv2.imwrite(vis_out_path, visual_mask)
        
        print(f"Traité : {base_name} ({len(features)} instances)")

    print("-" * 40 + "\nTraitement des instances terminé avec succès !")

# lancer ça pour split
def split_paired_dataset(image_dir, mask_dir, output_dir, img_ext, mask_ext, mask_suffix, train_ratio, val_ratio, test_ratio):
    
    total_ratio = train_ratio + val_ratio + test_ratio #  petite vérification des ratios, car on est tête en l'air tu connais 
    if not math.isclose(total_ratio, 1.0):
        print(f"Erreur : La somme des ratios ({total_ratio}) doit être égale à 1.0")
        return

  
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        print("Erreur : Le dossier des images ou le dossier des masques n'existe pas.")
        return

    # création des paires image, masque
    print("Image di r", image_dir)
    search_pattern = os.path.join(image_dir, f'*{img_ext}')
    print("search pattern",search_pattern)

    image_files = glob.glob(search_pattern)
    image_files.sort() # tri pour la reproductibilité
    
    paired_files = []
    print(image_files)
    for img_path in image_files:
        #maprint(img_path )
        # extraire le nom de base de l'image (ex: 'roi_099' à partir de 'roi_099.tif')
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # reprendre  le nom exact du masque correspondant
        mask_name = f"{base_name}{mask_suffix}{mask_ext}"
        mask_path = os.path.join(mask_dir, mask_name)
        
        # vérifie que le masque existe bien pour cette image
        if os.path.exists(mask_path):
            paired_files.append((img_path, mask_path))
        else:
            print(f"Masque introuvable pour l'image : {base_name}{img_ext} (Cherché: {mask_name})")

    total_pairs = len(paired_files)
    if total_pairs == 0:
        print("Erreur : Aucune paire Image/Masque correspondante n'a été trouvée.")
        return

    print(f"{total_pairs} paires Image/Masque trouvées. Mélange et répartition...\n" + "-"*40)

    # random.seed garantit que le mélange est reproductible
    random.seed(42) 
    random.shuffle(paired_files)

    # calcul des indices de coupe
    train_end = int(total_pairs * train_ratio)
    val_end = train_end + int(total_pairs * val_ratio)

    train_pairs = paired_files[:train_end]
    val_pairs = paired_files[train_end:val_end]
    test_pairs = paired_files[val_end:]

    print(f"Train : {len(train_pairs)} images/masques ({train_ratio*100}%)")
    print(f"Val : {len(val_pairs)} images/masques ({val_ratio*100}%)")
    print(f"Test : {len(test_pairs)} images/masques ({test_ratio*100}%)\n")

    #Fonction pour copier les paires dans les bons dossiers
    def copy_pairs(pair_list, split_name):
        dest_img_dir = os.path.join(output_dir, split_name, 'images')
        dest_mask_dir = os.path.join(output_dir, split_name, 'masks')
        
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_mask_dir, exist_ok=True)
        
        for img_path, mask_path in pair_list:
            shutil.copy2(img_path, os.path.join(dest_img_dir, os.path.basename(img_path)))
            shutil.copy2(mask_path, os.path.join(dest_mask_dir, os.path.basename(mask_path)))


    print("Copie des fichiers en cours (cela peut prendre quelques secondes)...")
    copy_pairs(train_pairs, 'train')
    copy_pairs(val_pairs, 'val')
    copy_pairs(test_pairs, 'test')

    print(f"\nLe  dataset complet est prêt dans '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sépare un dataset complet (Images + Masques) en Train/Val/Test."
    )
    
    parser.add_argument("-img", "--image_dir", required=True, help="Dossier contenant les images originales (H&E)")
    parser.add_argument("-msk", "--mask_dir", required=True, help="Dossier contenant les masques générés")
    parser.add_argument("-o", "--output_dir", default="dataset_ready", help="Dossier final (défaut: dataset_ready)")
    
    parser.add_argument("--img_ext", default=".tif", help="Extension des images originales (défaut: .tif)")
    parser.add_argument("--mask_ext", default=".tif", help="Extension des masques (défaut: .tif)")
    
    # Très important : permet au script de faire le lien entre "roi_099.tif" et "roi_099_mask.png"
    parser.add_argument("--mask_suffix", default="_mask", help="Suffixe ajouté au masque par rapport à l'image. Mettre '' si les noms sont identiques (défaut: '_mask')")
    
    parser.add_argument("--train", type=float, default=0.75, help="Ratio Train (défaut: 0.75)")
    parser.add_argument("--val", type=float, default=0.15, help="Ratio Validation (défaut: 0.15)")
    parser.add_argument("--test", type=float, default=0.10, help="Ratio Test (défaut: 0.10)")

    args = parser.parse_args()

    split_paired_dataset(
        args.image_dir, args.mask_dir, args.output_dir, 
        args.img_ext, args.mask_ext, args.mask_suffix, 
        args.train, args.val, args.test
    )