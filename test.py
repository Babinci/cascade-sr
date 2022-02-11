from math import log10, sqrt
from dataset import *
from FSRCNN import *
from utils import *


def PSNR(original, compressed):
    mse = torch.mean((original.float() - compressed.float()) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def normalize_back(img_tensor):
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    img_tensor = img_tensor * STD[:, None, None] + MEAN[:, None, None]
    img_tensor = (255 *img_tensor/torch.max(img_tensor)).to(torch.uint8)
    return img_tensor


# only set 5 x4 currently
images_list = sorted(load_images_list('../datasets/test/Set5_SR/Set5/image_SRF_4', ends='png'))

to_compare = ['HR','LR', 'SRCNN']
test_images = []
for file in to_compare:
    images_for_test = sorted([img for img in images_list if img[:-4].endswith(file)])
    test_images.append(images_for_test)
test_images = sorted(test_images)
final_list = [(a,b,c) for (a,b,c) in zip(test_images[0],test_images[1],test_images[2])]




def calculate_metrics(model, final_list, device):
    a = 0
    FSRCNN_metric_mean, model_metric_mean = 0,0
    for images in final_list:
        a+=1
        HR_image = torchvision.io.read_image(images[0])
        FSRCNN_image = torchvision.io.read_image(images[2])

        LR_image = torchvision.io.read_image(images[1]).to(torch.float)
        LR_image = normalize_img(LR_image).to(device)

        with torch.no_grad():
            upscaled = model.upscale_4(LR_image.unsqueeze(0)).squeeze(0)

        upscaled = normalize_back(upscaled.to('cpu'))

        FSRCNN_metric = PSNR(HR_image, FSRCNN_image)
        model_metric = PSNR(HR_image, upscaled)
        
        FSRCNN_metric_mean += FSRCNN_metric
        model_metric_mean += model_metric

    FSRCNN_metric_mean /= a
    model_metric_mean /= a

    return FSRCNN_metric_mean, model_metric_mean

