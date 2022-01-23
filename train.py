from torch import nn
import torch.optim as optim
from tqdm import tqdm

from dataset import *
from FSRCNN import *
from utils import *

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
print(os.listdir())
def main():
    dataset = Images("../datasets/", scale=8)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(len(dataset))
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
    model2 = Cascade2(base_model)
    # optimizer = optim.Adam(lr=1e-4)
    epochs = 1000

    for epoch in tqdm(range(epochs)):
        for img, resized_img2, resized_img4, resized_img8, _, _ in dataloader:

            pred0, pred1, pred2 = model2(resized_img8.to(device))

            # W, H = img.shape[-1], img.shape[-2]

            L0 = criterion(pred0, resized_img4.to(device))#/ ((0.25**2) *W*H)
            L1 = criterion(pred1, resized_img2.to(device))# / ((0.5**2) *W*H)
            L2 = criterion(pred2, img.to(device))# / (W*H)
            loss = L0 + L1 + L2
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # writer.add_scalar('Loss/train', loss, epoch)
if __name__ == "__main__":
    main()