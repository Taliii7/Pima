import os 
from PIL import Image 

class HistopathologyDataset : 
    def __init__(self,image_dir,mask_dir, transform=None):
        self.image_dir = image_dir 
        self.mask_dir = mask_dir
        self.transform =transform
        self.images = os.listdir(image_dir)

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx): 

        image_path = os.path.join(self.image_dir,self.images[idx] )
        mask_path= os.path.join(self.image_dir, self.images[idx])
      
        image = Image.open(image_path)
        mask = Image.open(mask_path)# l'étiquette


        # todo: Ajouter les transfos (redimensionnement, conversion en Tenseurs etc....)

        return image, mask 
    


        


print(sorted(os.listdir("./wallpaper")) ) 
os.path.join