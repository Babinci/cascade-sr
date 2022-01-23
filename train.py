from torch import nn
import torch.optim as optim
from tqdm import tqdm

from dataset import *
from FSRCNN import *
from utils import *

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def main():
    dataset = Images("../datasets/train/", scale=8)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    lr_cur = 1e-3  # current lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = FSRCNN().to(device)
   
    criterion = nn.L1Loss()

    optimizer = optim.Adam(
        [
            {"params": base_model.first_part.parameters()},
            {"params": base_model.mid_part.parameters()},
            {"params": base_model.last_part.parameters(), "lr": lr_cur * 0.1},
        ],
        lr=lr_cur,
    )
    model2 = Cascade(base_model)
    
    epochs = 1000

    for epoch in tqdm(range(epochs)):
        for img, resized_img2, resized_img4, resized_img8, _, _ in dataloader:

            pred0, pred1, pred2 = model2(resized_img8.to(device))

            
            L0 = criterion(pred0, resized_img4.to(device))
            L1 = criterion(pred1, resized_img2.to(device))
            L2 = criterion(pred2, img.to(device))
            loss = L0 + L1 + L2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #optimizer updating
    lr_cur = lr_cur * (0.1 ** int( epoch / (epochs * 0.8) ) )
    optimizer = optim.Adam(
        [
            {"params": base_model.first_part.parameters()},
            {"params": base_model.mid_part.parameters()},
            {"params": base_model.last_part.parameters(), "lr": lr_cur * 0.1},
        ],
        lr=lr_cur,
    )


    writer.add_scalar('Loss/train', loss, epoch)
if __name__ == "__main__":
    main()