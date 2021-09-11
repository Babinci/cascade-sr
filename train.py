from torch import nn
import torch.optim as optim
from tqdm import tqdm

from dataset import *
from FSRCNN import *
from utils import *


dataset = Images("../datasets", scale=8)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

lr_cur = 1e-3  # current lr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FSRCNN().to(device)
# model2 = Cascade()

# criterion = nn.MSELoss()
criterion = nn.L1Loss()

optimizer = optim.Adam(
    [
        {"params": model.first_part.parameters()},
        {"params": model.mid_part.parameters()},
        {"params": model.last_part.parameters(), "lr": lr_cur * 0.1},
    ],
    lr=lr_cur,
)
model2 = Cascade2(model)
# optimizer = optim.Adam(lr=1e-4)
epochs = 100

for epoch in tqdm(range(epochs)):
    for img, resized_img2, resized_img4, resized_img8, _, _ in dataloader:

        pred0, pred1, pred2 = model2(resized_img8)

        L0 = criterion(pred0, resized_img4)
        L1 = criterion(pred1, resized_img2)
        L2 = criterion(pred2, img)
        loss = L0 + L1 + L2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(loss)
model2._save_to_state_dict("model1.pth")
