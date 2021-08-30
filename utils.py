import torch
import os
print(torch)


def load_images_list(directory, ends='png'):
    img_list = [
        os.path.join(root,name)
        for root, dirs,files in os.walk(directory, topdown=False)
        for name in files
        if os.path.join(root, name).endswith(ends)
    ]
    return img_list