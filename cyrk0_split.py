import os
import shutil
import pandas as pd




# 1. Dictionnaire de tes configurations (Grossissement -> Liste des résolutions)
def directory_split(base_input_dir = "cytoDArk0",csv_dir ="folds",base_output_dir = "cytoDArk_split" ):
    configs = {
        "20x": ["256", "512", "1024"],
        "40x": ["256", "512", "1024", "2048"]
    }

    # Mappage des folds aux noms de dossiers
    fold_mapping = {
        0: "train",
        1: "val",
        2: "test"
    }

    print("Lancement du script de séparation des données...")

    # 3. Boucle sur les grossissements et résolutions
    for mag, res_list in configs.items():
        for res in res_list:
            res_folder = f"{res}x{res}" # Ex: 256x256
            csv_filename = os.path.join(csv_dir, f"folds{mag.replace('x', '')}_{res}.csv")
            
            if not os.path.exists(csv_filename):
                print(f"Fichier {csv_filename} introuvable. Ignoré.")
                continue
                
            print(f"\n==> Traitement de {mag} - {res_folder} avec {csv_filename}...")
            df = pd.read_csv(csv_filename)

            df = df.dropna(subset=['img_id', 'fold'])
            
            # Compteurs pour suivi
            compteurs = {"train": 0, "val": 0, "test": 0}
            
            for index, row in df.iterrows():
                img_id = str(row['img_id']).strip()
                fold = int(row['fold'])
                
                if fold not in fold_mapping:
                    continue
                    
                split_name = fold_mapping[fold]
                
                # Chemins d'origine 
                src_img = os.path.join(base_input_dir, mag, res_folder, "image", f"{img_id}.png")
                src_lbl = os.path.join(base_input_dir, mag, res_folder, "label", f"{img_id}.tiff")
                
                #Chemins de destination 
                dest_dir = os.path.join(base_output_dir, mag, res_folder, split_name)
                os.makedirs(dest_dir, exist_ok=True) # Crée le dossier s'il n'existe pas
                
                dest_img = os.path.join(dest_dir, f"{img_id}.png")
                #On ajoute _masks au label ground truth pour plus de lisibilité
                dest_lbl = os.path.join(dest_dir, f"{img_id}_masks.tiff") 
                
                # Copie sécurisée 
                if os.path.exists(src_img) and os.path.exists(src_lbl):
                    shutil.copy(src_img, dest_img)
                    shutil.copy(src_lbl, dest_lbl)
                    compteurs[split_name] += 1
                else:
                    print(f"[!] Attention, fichiers manquants pour l'ID: {img_id}")
            
            print(f"Fait ! (Train: {compteurs['train']}, Test: {compteurs['test']}, Val: {compteurs['val']})")

    print(f"\n Split terminé avec succès ! Les données sont prêtes pour Cellpose dans le dossier {base_output_dir} .")




def train_val_split(base_dir="cytoDArk_split"):
    train_images_paths = []
    train_masks_paths = []
    val_images_paths = []
    val_masks_paths = []

    print(f"Exploration du dossier {base_dir} pour trouver les données...")

    # os.walk parcourt absolument tous les sous-dossiers de l'arborescence
    for root, dirs, files in os.walk(base_dir):
        folder_name = os.path.basename(root)
        
        if folder_name in ['train', 'val']:
            for f in files:
                if f.endswith('.png'):
                    img_path = os.path.join(root, f)
                    # On va essayer de deviner le nom du masque correspondant
                    mask_path = os.path.join(root, f.replace('.png', '_masks.tiff'))
                    
                    if os.path.exists(mask_path):
                        if folder_name == 'train':
                            train_images_paths.append(img_path)
                            train_masks_paths.append(mask_path)
                        else: # val
                            val_images_paths.append(img_path)
                            val_masks_paths.append(mask_path)
    
    return train_images_paths, train_masks_paths, val_images_paths, val_masks_paths



if __name__ == "__main__":
    directory_split(base_input_dir = "cytoDArk0",csv_dir ="folds", base_output_dir = "cytoDArk_split")