from  cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
import cellvit.training.utils.metrics
import torch
import tifffile
import numpy as np
import os
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import evaluate_models
import utils

def load_model(checkpoint):

	checkpoint_info = torch.load(checkpoint)
	conf = checkpoint_info['config'] #??
	#checkpoint_data = torch.load(checkpoint, map_location="cpu")
	#print("keys = ", checkpoint_data.keys())
	model = CellViTSAM(model_path=None,
	num_nuclei_classes=conf["data.num_nuclei_classes"],
	num_tissue_classes=conf["data.num_tissue_classes"],
	vit_structure='SAM-H',
	drop_rate = conf['training.drop_rate'] #I'll have to check this, not sure but I remember seeing it in the config file
	)

	model.load_state_dict(checkpoint_info['model_state_dict'])
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	print("Loading model done")
	return model, device

def load_normalize_x(img_path, device):

	#img = Image.open("ID1_Aud_Cortex_Tursiops_1.png")
	img = Image.open(img_path)
	transform = T.Compose([
		T.ToTensor(),
		T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])
	img_tensor = transform(img).unsqueeze(0).to(device)

	return img_tensor


def test_model(model, device, x_folder, output_folder, mag):

	#maps = inference_on_x(model, x, mag)
	x_paths = sorted([os.path.join(x_folder, f) for f in os.listdir(x_folder) if f.endswith('.png')])
	#res = []
	shutil.rmtree('../out/', ignore_errors=True); os.makedirs('../out/', exist_ok=True)
	results = []
	for p in x_paths:
		img_tensor = load_normalize_x(p, device)
		with torch.no_grad():
			outputs = model(img_tensor)
			instance_map, _ = model.calculate_instance_map(outputs, magnification=mag)
		pred_mask = instance_map[0]
		print(np.unique(pred_mask))

		name = p.split('/')[-1]
		out = output_folder + name
		#img = pred_mask.cpu().numpy().astype('uint8') - 1 #test, because different indices, don't understand and pretty pissed
		img = Image.fromarray(pred_mask.cpu().numpy().astype('uint8'))
		#img = Image.fromarray(img)
		img.save(out)
		print(f"Done  for image {p}")
		y_path = p.replace('.png', '_masks.tiff')
		y_path = y_path.replace('image', 'label')
		if os.path.exists(y_path):
		#	mask = tifffile.imread(y_path)
			mask = utils.imread(y_path)
			_, _, infos, _ = evaluate_models.get_pq(mask, pred_mask, 0.5, remap=True)
			paired_true, paired_pred, unpaired_pred, unpaired_true = infos #[0], infos[1], infos[2], infos[3]
			scores = evaluate_models.cell_detection_scores(paired_true, paired_pred, unpaired_true, unpaired_pred, w=[1, 1])
			results.append(scores)
		else:
			print(f"Not found : {y_path}")
		#	continue
		#res.append((pred_mask, mask))
		print(f"Done for {p}")
	return results, x_folder, output_folder

def main():

	x_folder = "../cytoDArk_split/40x/256x256/test"
	y_folder = "../cytoDArk_split/40x/256x256/test"
	output_folder = "../out/"
	checkpoint = "../CellViT-SAM-H-x40-AMP.pth"
	current_dir = os.path.dirname(os.path.abspath(__file__))
	json_path = os.path.join(current_dir, "", "json_cfg.json")

	#json_path = "~/json_cfg.json"

	model, device = load_model(checkpoint)
	results, _, _ = test_model(model, device, x_folder, output_folder, 20)
	print(f"Results : {results}")

if __name__ == '__main__':

	main()
