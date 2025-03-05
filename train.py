import torch
from model import centernet
import numpy as np
from tqdm import tqdm
from loss import centerloss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = centernet() 
model.to(device)



# Optimizer
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-4)

from dataset import CenternetDataset    

dataset = CenternetDataset("../../yolo_dataset/images/train","../../yolo_dataset/labels/train") 
train_loader = torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=True, num_workers=0)

def train(epoch):
    model.train()
    running_loss = 0.0
    running_mask = 0.0
    running_regr = 0.0
    t = tqdm(train_loader)
    rd = np.random.rand()
    print(epoch)
    for idx, (img, hm, regr) in enumerate(t):       
        # send to gpu
        img = img.to(device)
        hm_gt = hm.to(device)
        regr_gt = regr.to(device)
        # set opt
        optimizer.zero_grad()
        
        # run model
        hm, regr = model(img)
        preds = torch.cat((hm, regr), 1)
            
        loss, mask_loss, regr_loss = centerloss(preds, hm_gt, regr_gt)
        # misc
        running_loss += loss
        running_mask += mask_loss
        running_regr += regr_loss
        loss.backward()
        optimizer.step()
        print(f"running_loss : {running_loss/(idx+1)}")
        print(f"running_mask : {running_mask/(idx+1)}")
        print(f"running_regr : {running_regr/(idx+1)}")
        



for i in range(100):
    train(i)

torch.save(model, "entire_model.pt")
