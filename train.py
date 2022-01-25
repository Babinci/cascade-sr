from torch import nn
import torch.optim as optim
from tqdm import tqdm

from dataset import *
from FSRCNN import *
from utils import *

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

torch.manual_seed(0)

def main():
    init_dataset = Images("../datasets/train/", scale=8)
    train_dataset, val_dataset = torch.utils.data.dataset.random_split(init_dataset, [172, 19])
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
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
    model = Cascade(base_model)
    
    epochs = 250

    i = 0

    for epoch in tqdm(range(epochs)):
        #train loop
        model.train()
        for img, resized_img2, resized_img4, resized_img8, _, _ in train_dataloader:

            pred0, pred1, pred2 = model(resized_img8.to(device))

            L0 = criterion(pred0, resized_img4.to(device))
            L1 = criterion(pred1, resized_img2.to(device))
            L2 = criterion(pred2, img.to(device))
            loss = L0 + L1 + L2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #validation
        model.eval()
        with torch.no_grad():
            val_loss = []
            for img, resized_img2, resized_img4, resized_img8, _, _ in val_dataloader:

                pred0, pred1, pred2 = model(resized_img8.to(device))

                L0 = criterion(pred0, resized_img4.to(device))
                L1 = criterion(pred1, resized_img2.to(device))
                L2 = criterion(pred2, img.to(device))

                loss = L0 + L1 + L2
                val_loss.append(loss)
            avg_val_loss = torch.stack(val_loss).mean()
            writer.add_scalar('val_loss', avg_val_loss, epoch)

        #optimizer updating as in paper
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * (0.1 ** int( epoch / (epochs * 0.8) ) )
        
        # path = f'../model_chkpoints/{i}.pth'
        # torch.save(model2.state_dict(), path)
        # i +=1 
        

if __name__ == "__main__":
    main()