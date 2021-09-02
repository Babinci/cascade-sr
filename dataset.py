from utils import *
from torch.utils.data import Dataset
import torchvision
from torch.utils.data import DataLoader

class Images(Dataset):
    def __init__(self, img_dir, ends='.png'):
        self.images = load_images_list(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = torchvision.io.read_image(img_path)
        return img, img_path
