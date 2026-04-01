import os
import argparse
import torch
import numpy as np
import tifffile

from monai.transforms import (
    Compose,
    EnsureTyped,
    ScaleIntensityd,
    ScaleIntensityRangePercentilesd
)
from monai.data import Dataset, DataLoader
from monai.inferers import SlidingWindowInfererAdapt

# Importation des modules spécifiques à VISTA-2D depuis leurs fichiers sources
try:
    from components import LoadTiffd, LogitsToLabels
    from cell_sam_wrapper import CellSamWrapper
except ImportError:
    from scripts.components import LoadTiffd, LogitsToLabels
    from scripts.cell_sam_wrapper import CellSamWrapper

def deviceChoice():
    """
    Selectionne le meilleur device disponible.
    """
    if torch.cuda.is_available():
        print("Super ! Le GPU NVIDIA est activé !")
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        print("Le GPU du Mac M1/M2/M3 (MPS) est activé.")
        return torch.device("mps")
    else:
        print("Attention, on tourne sur le CPU.")
        return torch.device("cpu")

def load_vista_model(sam_ckpt, vista_ckpt, device):
    """Charge l'architecture VISTA-2D et ses poids
    
    """
    print("Construction de l'architecture VISTA-2D (SAM)...")
    model = CellSamWrapper(checkpoint=sam_ckpt)
    model.to(device)
    
    print(f"Chargement des poids VISTA depuis : {vista_ckpt}")
    checkpoint = torch.load(vista_ckpt, map_location=device, weights_only=True)
    
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def get_preprocessing_transforms():
    """
    Reproduit exactement les étapes de preprocessing du fochier json de VISTA
    """
    return Compose([
        LoadTiffd(keys=["image"]),
        EnsureTyped(keys=["image"], data_type="tensor", dtype=torch.float),
        ScaleIntensityd(keys=["image"], minv=0, maxv=1, channel_wise=True),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, channel_wise=True, clip=True
        )
    ])

def main():

    # Pour rendre le script paramétrisable 
    parser = argparse.ArgumentParser(description="Inférence VISTA-2D simplifiée")
    parser.add_argument("--image", type=str, required=True, help="Chemin vers l'image (.png, .tif)")
    parser.add_argument("--out", type=str, default="resultat_vista.tif", help="Nom du fichier de sortie")
    parser.add_argument("--sam_ckpt", type=str, default="models/sam_vit_b_01ec64.pth", help="Poids SAM")
    parser.add_argument("--vista_ckpt", type=str, default="models/model.pt", help="Poids VISTA")
    
    args = parser.parse_args()


    # Va choisir le device le plus approprié pour l'inference ()
    device = deviceChoice()
    

    #1 Preparation de l'inference 


    ## on charge le model
    model = load_vista_model(args.sam_ckpt, args.vista_ckpt, device)
    
    # post process de l'image, (si on observe des meilleurs resultat sur Vista2 c'est possible que ce soit pas que grâce aux nouveaux poids mais aussi à cette étape de post processing )
    print(f"Chargement et prétraitement de l'image : {args.image}")
    transforms = get_preprocessing_transforms()
    
    # 
    data_dict = {"image": args.image}
    dataset = Dataset(data=[data_dict], transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # On reprends les mêmes paramtres que dans le fichier json
    inferer = SlidingWindowInfererAdapt(
        roi_size=[256, 256],
        sw_batch_size=1,
        overlap=0.625,
        mode="gaussian",
        cache_roi_weight_map=True,
        progress=False
    )
    
    # 2 Inference 
    print("Segmentation en cours (Sliding Window)...")
    post_processor = LogitsToLabels() # L'outil officiel de NVIDIA basé sur Cellpose !
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["image"].to(device)
            
            # (vient du fichier inference.json à la base) utilisation de l'AMP (Automatic Mixed Precision) uniquement sur GPU NVIDIA pour accélérer
            use_amp = device.type == "cuda"
            amp_dtype = torch.float16 if use_amp else torch.float32
            
            # ça sert à optimiser la vitesse de calcul, ils divisne tles float 32 (précis mais lourd) en float 16 pour seulement certaines multiplications 
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                logits = inferer(inputs=inputs, network=model)
            
            # on applique le pré processing
            print("Calcul de la dynamique des flux...")
            # forme des logits  :  [Batch, Channels, Height, Width]. On prend le premier élément du batch.
            logits_b0 = logits[0] 
            
            # LogitsToLabels renvoie le masque et les probabilités , on garde que le masque.
            pred_mask, _ = post_processor(logits_b0, filename=args.image) 
            
            print(f"Terminé ! {pred_mask.max()} cellules trouvées.")
            
            # Sauvegarde, je mets le fichier tiff histoire de l'érire en dur pour récupérer pour des futurs tests plus tard
            final_mask = pred_mask.astype(np.uint16)
            tifffile.imwrite(args.out, final_mask)
            print(f"Masque sauvegardé sous : {args.out}")

if __name__ == "__main__":
    main()