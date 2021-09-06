from torch import nn
import torch.optim as optim
from tqdm import tqdm

from dataset import *
from FSRCNN import *
from utils import *



dataset = Images('../datasets', scale = .5)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

lr_cur =1e-3 #current lr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = FSRCNN().to(device)



criterion = nn.MSELoss()
optimizer = optim.Adam([
    {'params': model.first_part.parameters()},
    {'params': model.mid_part.parameters()},
    {'params': model.last_part.parameters(), 'lr': lr_cur * 0.1}
], lr=lr_cur)

epochs = 100

for epoch in tqdm(range(epochs)):
    for img, resized_img,_,_ in dataloader:

        preds = model(resized_img)
        loss = criterion(preds, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(loss)

