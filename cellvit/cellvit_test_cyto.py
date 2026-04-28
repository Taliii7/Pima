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
import csv

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


def test_model(model, device, x_folder, output_folder, mag, iou):

	#maps = inference_on_x(model, x, mag)
	x_paths = sorted([os.path.join(x_folder, f) for f in os.listdir(x_folder) if f.endswith('.png')])
	#res = []
	#shutil.rmtree(output_folder, ignore_errors=True)
	#os.makedirs('output_folder', exist_ok=True)
	results = []
	for p in x_paths:
		img_tensor = load_normalize_x(p, device)
		with torch.no_grad():
			outputs = model(img_tensor)
			instance_map, _ = model.calculate_instance_map(outputs, magnification=mag)
		pred_mask = instance_map[0]
		print(np.unique(pred_mask))

		name = p.split('/')[-1]
		out = os.path.join(output_folder, name)
		#img = pred_mask.cpu().numpy().astype('uint8') - 1 #test, because different indices, don't understand and pretty pissed
		img = Image.fromarray(pred_mask.cpu().numpy().astype('uint8'))
		#img = Image.fromarray(img)
		img.save(out)
		#print(f"Done  for image {p}")
		y_path = p.replace('.png', '_masks.tiff')
		y_path = y_path.replace('image', 'label')
		if os.path.exists(y_path):
		#	mask = tifffile.imread(y_path)
			mask = utils.imread(y_path)
			_, _, infos, _ = evaluate_models.get_pq(mask, pred_mask, iou, remap=True)
			paired_true, paired_pred, unpaired_pred, unpaired_true = infos #[0], infos[1], infos[2], infos[3]
			scores = evaluate_models.cell_detection_scores(paired_true, paired_pred, unpaired_true, unpaired_pred, w=[1, 1])
			results.append(scores)
		else:
			print(f"Not found : {y_path}")
		#	continue
		#res.append((pred_mask, mask))
		print(f"Done for {p}")
	filepath = os.path.join(output_folder, "results.csv")
	with open(filepath, "w") as fout:
		csv_out = csv.writer(fout)
		csv_out.writerow(["f1", "precision", "recall"])
		for row in results:
			csv_out.writerow(row)
	return results, x_folder, output_folder

def main():

	x_folder = "../cytoDArk_split/40x/256x256/test"
	y_folder = "../cytoDArk_split/40x/256x256/test"
	output_folder_20x_0_5 = "../out_20x_0_5"
	output_folder_20x_0_6 = "../out_20x_0_6"
	x_folder_20 = "../cytoDArk_split/20x/256x256/test"
	y_folder_20 = "../cytoDArk_split/20x/256x256/test"

	output_folder_40x_0_5 = "../out_40x_0_5"
	output_folder_40x_0_6 = "../out_40x_0_6"
	x_folder_40 = "../cytoDArk_split/40x/256x256/test"
	y_folder_40 = "../cytoDArk_split/40x/256x256/test"

	folders = [(output_folder_20x_0_5, x_folder_20, y_folder_20, 20, 0.5), 
		(output_folder_20x_0_6, x_folder_20, y_folder_20, 20, 0.6), 
		(output_folder_40x_0_5, x_folder_40, y_folder_40, 40, 0.5), 
		(output_folder_40x_0_6, x_folder_20, y_folder_20, 40, 0.6)]
	checkpoint = "../CellViT-SAM-H-x40-AMP.pth"

	model, device = load_model(checkpoint)
	for out, x, y, mag, iou in folders:
		results, _, _ = test_model(model, device, x, out, mag, iou)
		print(f"Results : {results}")

	print("output_folder_20x_0_5/results.csv")
	evaluate_models.get_mean("../out_20x_0_5/results.csv")

	print("output_folder_20x_0_6/results.csv")
	evaluate_models.get_mean("../out_20x_0_6/results.csv")

	print("output_folder_40x_0_5/results.csv")
	evaluate_models.get_mean("../out_40x_0_5/results.csv")

	print("output_folder_40x_0_6/results.csv")
	evaluate_models.get_mean("../out_40x_0_6/results.csv")

if __name__ == '__main__':

	main()
