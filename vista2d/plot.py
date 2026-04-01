import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from cellpose import utils, plot  # On importe les outils surpuissants de Cellpose !

def plot_segmentation(chemin_image, chemin_masque, chemin_sauvegarde=None):
    # 1. Charger l'image et la convertir en tableau de pixels standard
    img_pil = Image.open(chemin_image).convert("RGB")
    img0 = np.array(img_pil)
    
    # 2. Charger le masque VISTA-2D
    maski = tifffile.imread(chemin_masque)

    # 3. Préparer une grande figure avec 3 "sous-fenêtres" (subplots)
    fig = plt.figure(figsize=(18, 6))

    # --- Panneau 1 : L'image originale ---
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img0)
    ax.set_title("1. Image Originale")
    ax.axis("off")

    # --- Panneau 2 : Les contours (Inspiré de votre exemple) ---
    # Cette ligne fait tout le calcul mathématique instantanément !
    outlines = utils.masks_to_outlines(maski) 
    
    imgout = img0.copy()
    outX, outY = np.nonzero(outlines)
    imgout[outX, outY] = np.array([255, 0, 0])  # On colorie les bords en rouge pur
    
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(imgout)
    ax.set_title("2. Contours Prédits (Rouge)")
    ax.axis("off")

    # --- Panneau 3 : L'overlay (Superposition colorée) ---
    # Cellpose génère l'overlay avec des couleurs parfaites d'un seul coup
    overlay = plot.mask_overlay(img0, maski)
    
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(overlay)
    ax.set_title("3. Masques Superposés")
    ax.axis("off")

    # 4. Sauvegarder et afficher
    if chemin_sauvegarde:
        plt.savefig(chemin_sauvegarde, bbox_inches='tight', dpi=300)
        print(f"L'image a été sauvegardée ici : {chemin_sauvegarde}")

    plt.show()

# --- On lance la machine ! ---
image_png = "cellpose_dataset/ID1_Aud_Cortex_Tursiops_1.png"
masque_tif = "eval/mon_dauphin.tif"
plot_segmentation(image_png, masque_tif, "eval/resultat_dauphin.png")