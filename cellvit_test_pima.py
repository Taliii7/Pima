import torch
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import tifffile
from sklearn.metrics import  precision_score, recall_score, f1_score, jaccard_score
import os

def load_normalize_x(img_path, device):
    
    #img = Image.open("ID1_Aud_Cortex_Tursiops_1.png")
    img = Image.open(img_path)
    transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    return img_tensor

def load_model(checkpoint_path)
    
    #checkpoint_path = "CellViT-SAM-H-x40-AMP.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    conf = checkpoint["config"]

    model = CellViTSAM(
        model_path=None,
        num_nuclei_classes=conf["data.num_nuclei_classes"],
        num_tissue_classes=conf["data.num_tissue_classes"],
        vit_structure="sam-h"
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, device
    
def test_model(model, device, x_folder, y_folder, mag, threshold):

    maps = inference_on_x(model, x, mag)
    x_paths = sorted([os.path.join(x_folder, f) for f in os.listdir(x_folder) if f.endswith('.png')])
    
    TP = []
    #TN = []
    #FP = []
    #FN = []
    
    for p in x_paths:
        img_tensor = load_normalize_x(p, device)
        with torch.no_grad():
            outputs = model(img_tensor)
            instance_map, _ = model.calculate_instance_map(outputs, magnification=mag)
            pred_mask = (instance_map[0] > 0).astype(np.uint8)

        y_path = p.replace('.png', '_masks.tiff')
        if os.path.exists(y_path):
            mask = (tifffile.imread(y_path) > 0).astype(np.uint8)
        else:
            print(f"Not found : {y_path}")
            continue
        
        #My version not optimized but I am still proud of my work so I let it here
        #ones_pred = np.nonzero(pred_mask.flatten())
        #ones_truth = np.nonzero(mask.flatten())
        #inter = np.intersect1d(ones_pred, ones_truth)
        #IoU = (len(inter)/(len(ones_pred)+len(ones_truth) - len(inter)))
        
        inter = np.logical_and(pred_mask, mask).sum()
        uni = np.logical_or(pred_mask, mask).sum()
        IoU = inter/uni if uni > 0 else 0
        
        if IoU >= threshold:
            TP.append(IoU)
        print(f"Done  for image : {p}")
    print("All done")
    return TP
    
def main():
    
    checkpoint_path = "CellViT-SAM-H-x40-AMP.pth"
    x_folder = ""
    y_folder = ""
    mag = 20
    threshold = 0.6
    
    model, device = load_model(checkpoint_path)
    TPs = test_model(model, device, x_folder, y_folder, mag, threshold)

if __name__ == "__main__":

    main()
       
"""
imgs = [img, instance_map[0], label_data]
names = ["image", "prediction", "mask"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, img, name in zip(axes, imgs, names):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(name)
plt.tight_layout()
plt.show()
"""