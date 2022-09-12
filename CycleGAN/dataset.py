from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self,root_monet, root_photo, transform = None):
        self.root_monet = root_monet
        self.root_photo = root_photo
        self.transform = transform

        self.monet_images = os.listdir(root_monet)
        self.photo_images = os.listdir(root_photo)
        
        self.length_dataset = max(len(self.monet_images),len(self.photo_images))
        
        self.monet_len = len(self.monet_images)
        self.photo_len = len(self.photo_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        monet_img = self.monet_images[index%self.monet_len]
        photo_img = self.photo_images[index%self.photo_len]

        monet_path = os.path.join(self.root_monet,monet_img)
        photo_path = os.path.join(self.root_photo,photo_img)

        monet_img = np.array(Image.open(monet_path).convert("RGB"))
        photo_img = np.array(Image.open(photo_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=monet_img,image0=photo_img)
            monet_img = augmentations["image"]
            photo_img = augmentations["image0"]
            
        return monet_img,photo_img