from utils import *
from torch.utils.data import Dataset
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F 


def resize_img(img, scale =.5):
    return F.interpolate(img.unsqueeze(0), scale_factor=(scale,scale)).squeeze(0)

class Images(Dataset):
    def __init__(self, img_dir, ends='.png'):
        self.images = load_images_list(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = torchvision.io.read_image(img_path)
        resized_img = resize_img(img)
        return img, resized_img, img_path, index
