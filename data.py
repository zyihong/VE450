from torch.utils.data import Dataset
import os
from resizeimage import resizeimage
from PIL import Image
from tqdm import tqdm
import torch
import cv2
RESIZE_IMAGE_DIR="./resize"
IMAGE_DIR="./pictures"
IMAGE_SIZE=[64,64]

class Traindataset(Dataset):
    def __init__(self):
        images = os.listdir(IMAGE_DIR)
        X=[]
        Y=[]
        for i in images:
            X.append(torch.Tensor(cv2.imread(os.path.join(RESIZE_IMAGE_DIR,i))))
            Y.append(0 if i[0]=='A' else 1 )
        self.X=X
        self.Y=torch.Tensor(Y)
        self.len=self.Y.shape[0]

    def __getitem__(self,index):
        return self.X[index].permute((2,0,1)),self.Y[index]

    def __len__(self):
        return self.len



def resize_image():
    if not os.path.exists(RESIZE_IMAGE_DIR):
        os.makedirs(RESIZE_IMAGE_DIR)

    images = os.listdir(IMAGE_DIR)
    # num_images = len(images)

    for i, image in enumerate(tqdm(images)):
        with open(os.path.join(IMAGE_DIR, image), 'r+b') as f:
            with Image.open(f) as img:
                # img = resize_image(img, IMAGE_SIZE)
                resize_img = resizeimage.resize_cover(img, IMAGE_SIZE, validate=False)
                resize_img.save(os.path.join(RESIZE_IMAGE_DIR, image), img.format)

    print('Finish resizing images')

# resize_image()