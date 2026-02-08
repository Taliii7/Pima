import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor,  SamAutomaticMaskGenerator





def deviceAvailable():
    # méthode afin de savoir quel type de gpu est sur l'ordinateur, mps = la puce gpu du mac (metal silicon) 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"--- Utilisation du processeur : {device} ---")

    return device






def run_sam_inference(image_folder="images", output_folder="results"):
    # 1. Configuration du device (Spécifique Mac M1/M2/M3)
    device = deviceAvailable()

    # 2. Paramètres du modèle
    # Assure-toi que ce fichier est dans ton dossier actuel, je vais pas le commmit car le fichier est trop lourd
    sam_checkpoint = "sam_vit_b_01ec64.pth" 
    model_type = "vit_b" # c'est le model de transformer qui est compatible avec nos macs, vit_h est meilleur de ce que j'ai lu mais je peux pas le tester pour le moment 
    
    if not os.path.exists(sam_checkpoint):
        print(f"Erreur : {sam_checkpoint} introuvable. Télécharge-le d'abord !")
        return

    # 3. Chargement du modèle
    print("Chargement du modèle en mémoire...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    # Préparation des dossiers 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"Aucune image trouvée dans le dossier '{image_folder}'")
        return

    all_masks = []

    # 5. Boucle de traitement avec tqdm, tqdm sert simplement à afficher  dans le terminal là où en est le programme dans son execution, c'est pratique pour différentier les bugs des simples executions à rallonge
    for filename in tqdm(image_files, desc="Segmentation en cours"):
        # En gros ici c'est une boucle sur toutes les images du répertoire mis en paramètre, où tu vas placer un point pour segmenter l'objet qui y est affilié 
        print("IMAGE : ",filename)
        try:
            # On charge l'image
            path = os.path.join(image_folder, filename)
            image = cv2.imread(path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.imshow(image_rgb)
            
            # Calcul de l'embedding (le gros du travail sur M1)

            predictor.set_image(image_rgb)

            h, w, _ = image_rgb.shape #utile pour afficher rapidement les coordonées
            print("--------Coordonnées du point --------")
    
            X = float(input("X = "))
            Y = float (input("Y = "))
            # Si t'as pas d'idées de point à mettre, je t'ai mis les coordonnées que j'ai testé pour l'image que je t'ai envoyé sur discord en dessous 
            # si tu veux placer le point dans une zone precis, je te conseil de faire un plt.imshow de l'image de base, normalement t'aura les coordonés du point que tu clique en faisant ça 

            #220.9, 269.2 # clique sur le coeur de la cellule, sgmente uniquement le coeur
            #201.2, 277.9 clqiue sur la partie verte de la cellule, segmente toute la cellule
            input_point = np.array([[X, Y]])
            input_label = np.array([1])
 
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )

            #  Ici on garde le masque avec le meilleur score pour la liste finale (peut être changer pour obtenir les 3 masks pour un objet)
            best_mask_idx = np.argmax(scores)
            all_masks.append(masks[best_mask_idx])

            # 6. Sauvegarde visuelle du résultat
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            
            # Superposition du masque avec l'image pour plus de lisibilté  (violet transparente)
            mask_overlay = np.zeros((*masks[best_mask_idx].shape, 4))
            mask_overlay[masks[best_mask_idx]] = [1, 0, 1, 0.4] # Bleu avec 40% d'opacité, ça se modife dans l'ordre suivant [ R,G,B,Opacité]
            
            plt.imshow(mask_overlay)
            plt.scatter(input_point[:, 0], input_point[:, 1], color='red', marker='*', s=100, label='Prompt')
            plt.title(f"Score: {scores[best_mask_idx]:.2f}")
            plt.axis('off')
            
            # enregistre dans le répertoire mis en paramètre, si aucun répertoire de réponse n'est précisé, ça l'enregistrera dans le reperoire results par défaut 
            plt.savefig(os.path.join(output_folder, f"res_{filename}"))
            plt.close() 

        except Exception as e:
            print(f"Erreur sur l'image {filename}: {e}")

    print(f"\n--- Terminé.   |  Résultats dans '{output_folder}' --")
    return all_masks

def segmentAll(image_path): # la segmentation automatique est bien plus longue, c'est plus partique de tester imaege par image
    sam_checkpoint = "sam_vit_b_01ec64.pth" 
    model_type = "vit_b"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("IMAGE : ",image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # SAM travaille en RGB

    # Envoyer les données de l'image (le tableau numpy)
    masks = mask_generator.generate(image) # je sais pas si on pourrait utiliser les 3 masks par objet en segmentation automatique, à tester plus tard
    

    # affichage
    plt.figure(figsize=(10, 10))
    plt.imshow(image) 
    show_anns(masks)  # Méthode chiante pour superposer les masques sur l'image
    plt.axis('off')
    plt.show()

    return masks


def show_anns(anns):
    # méthode pour superposer les masques graphiquement, pas vraiment d'intérêt à la compréhension 
    if len(anns) == 0:
        return
    # Trier par surface (area) pour afficher les petits masques par desssu les grandes
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        # on mets une couleur aléatoire (RGB)
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        
        # On modife l'opacité du mask pour que ça reste lisible avec la superposition
        ax.imshow(np.dstack((img, m * 0.35)))


