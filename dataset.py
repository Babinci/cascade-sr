from torch._C import set_autocast_enabled
from utils import *
from torch.utils.data import Dataset
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F


def resize_img(img, scale=2):
    scale = 1/scale
    return F.interpolate(img.unsqueeze(0), scale_factor=(scale, scale)).squeeze(0)


def crop_img(img, mult=2):
    if img.shape[1] % mult != 0 and img.shape[2] % mult != 0:
        img = img[:, : -(img.shape[1] % mult), : -(img.shape[2] % mult)]
    elif img.shape[1] % mult != 0:
        img = img[:, : -(img.shape[1] % mult), :]
    elif img.shape[2] % mult != 0:
        img = img[:, :, : -(img.shape[2] % mult)]

    return img


class Images(Dataset):
    def __init__(self, img_dir, ends=".png", scale=8):
        self.images = load_images_list(img_dir, ends=ends)
        self.scale = scale

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = torchvision.io.read_image(img_path)
        # print(img.shape)
        img = crop_img(img, mult = self.scale).to(torch.float)
        # print(img.shape)
        resized_img2 = resize_img(img, scale=2).to(torch.float)
        resized_img4 = resize_img(img, scale=4).to(torch.float)
        resized_img8 = resize_img(img, scale=8).to(torch.float)
        return img, resized_img2, resized_img4, resized_img8, img_path, index
