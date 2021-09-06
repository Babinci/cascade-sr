from torch._C import set_autocast_enabled
from utils import *
from torch.utils.data import Dataset
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F



def resize_img(img, scale=0.5):
    return F.interpolate(img.unsqueeze(0), scale_factor=(scale, scale)).squeeze(0)

def crop_img(img, mult=2):
    if img.shape[1] % mult != 0 and img.shape[2] % mult != 0:
        img = img[:,:-1,:-1]
    elif img.shape[1] % mult != 0:
        img = img[:,:-1,:]
    elif img.shape[2] % mult != 0:
        img = img[:,:,:-1]
    
    return img


class Images(Dataset):
    def __init__(self, img_dir, ends=".png", scale =.5):
        self.images = load_images_list(img_dir)
        self.scale = scale
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = torchvision.io.read_image(img_path)
        # print(img.shape)
        img = crop_img(img)
        # print(img.shape)
        resized_img = resize_img(img, scale=self.scale)
        return img.to(torch.float), resized_img.to(torch.float), img_path, index
