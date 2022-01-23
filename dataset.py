from torch._C import set_autocast_enabled
from utils import *
from torch.utils.data import Dataset
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF


def resize_img(img, scale=2):
    scale = 1 / scale
    return F.interpolate(img.unsqueeze(0), scale_factor=(scale, scale)).squeeze(0)


def scale_img(img, p=0.8):
    """for augmentation"""
    scaling_factors = [0.9, 0.8, 0.7, 0.6]
    choice = random.choice(scaling_factors)
    if random.uniform(0, 1) < 0.8:
        return resize_img(img, scale=1 / choice)
    else:
        return img


def rotate_img(img, p=0.75):
    angles = [90, 180, 270]
    angle = random.choice(angles)
    if random.uniform(0, 1) < p:
        return TF.rotate(img, angle)
    else:
        return img


def crop_img(img, mult=2):
    if img.shape[1] % mult != 0 and img.shape[2] % mult != 0:
        img = img[:, : -(img.shape[1] % mult), : -(img.shape[2] % mult)]
    elif img.shape[1] % mult != 0:
        img = img[:, : -(img.shape[1] % mult), :]
    elif img.shape[2] % mult != 0:
        img = img[:, :, : -(img.shape[2] % mult)]

    return img


def normalize_img(img):
    normalize = torchvision.transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    return normalize(img / 255)


def augment_img(img):
    pass


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
        img = normalize_img(img)
        img = scale_img(img)
        img = rotate_img(img)
        img = crop_img(img, mult=self.scale).to(torch.float)

        # print(img.shape)
        resized_img2 = resize_img(img, scale=2).to(torch.float)
        resized_img4 = resize_img(img, scale=4).to(torch.float)
        resized_img8 = resize_img(img, scale=8).to(torch.float)
        return img, resized_img2, resized_img4, resized_img8, img_path, index


#dataset test
def main():
    dataset = Images("../datasets", scale=8)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    for img, resized_img2, resized_img4, resized_img8, _, _ in dataloader:
        print(img.shape)


if __name__ == "__main__":
    main()