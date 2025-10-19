# ----------------------------------------------------------------------
# Author: Adapted by ChatGPT
# HiFi-IFDL Training Script (CSV Dataset)
# ----------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from IMD_dataloader import train_dataset_loader_init
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_cfg_defaults
from models.NLCDetection_loc import NLCDetection
from utils.custom_loss import IsolatingLossFunction, load_center_radius
from tqdm import tqdm

device = torch.device('cuda:0')
device_ids = [0, 1]  # 使用 GPU 0 和 1

# -------------------------
# Training function
# -------------------------
def train(args, train_loader, FENet, SegNet, LOSS_MAP, num_epochs=3):
    FENet.train()
    SegNet.train()

    optimizer = optim.Adam(list(FENet.parameters()) + list(SegNet.parameters()), lr=args['lr'])

    for epoch in range(num_epochs):
        epoch_loss = 0
        # tqdm 进度条包装 train_loader
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for step, (images, masks, labels, paths) in loop:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            output = FENet(images)
            mask1_fea, mask_binary, out0, out1, out2, out3 = SegNet(output, images)

            if args['loss_type'] == 'dm':
                loss_map, loss_manip, loss_nat = LOSS_MAP(mask1_fea, masks)
                loss = loss_map + loss_manip + loss_nat
            else:
                mask_binary = mask_binary.float()
                masks = masks.float()
                loss = nn.BCELoss()(mask_binary.unsqueeze(1), masks.unsqueeze(1))

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # 更新 tqdm 描述
            loop.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg_Loss': f'{epoch_loss/(step+1):.4f}'})

        print(f"Epoch {epoch+1} Finished, Avg Loss: {epoch_loss/len(train_loader):.4f}")

        os.makedirs(args['weight_dir'], exist_ok=True)
        torch.save({
            'model_FENet': FENet.state_dict(),
            'model_SegNet': SegNet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1
        }, os.path.join(args['weight_dir'], f"epoch_{epoch+1}.pth"))
        print(f"Saved weights to {args['weight_dir']}/epoch_{epoch+1}.pth")

# -------------------------
# Main
# -------------------------
def main():
    args = {
        'csv_file': '/data/disk2/yer/Dataset/train/train.csv',
        'img_dir': '/data/disk2/yer/Dataset/train',
        'weight_dir': '/data/disk2/yer/ASOTA/HiFi_IFDL/weightpoints',
        'lr': 5e-5,
        'loss_type': 'ce'
    }

    train_loader = train_dataset_loader_init(args['csv_file'], args['img_dir'], batch_size=16)

    # Load models
    FENet_cfg = get_cfg_defaults()
    FENet = get_seg_model(FENet_cfg).to(device)
    SegNet = NLCDetection().to(device)

    FENet = nn.DataParallel(FENet, device_ids=device_ids).to(device)
    SegNet = nn.DataParallel(SegNet, device_ids=device_ids).to(device)

    # Load center & radius for IsolatingLossFunction
    center, radius = load_center_radius(args, FENet, SegNet, train_data_loader=train_loader, center_radius_dir='./center_loc')
    LOSS_MAP = IsolatingLossFunction(center, radius).to(device)

    train(args, train_loader, FENet, SegNet, LOSS_MAP, num_epochs=100)

if __name__ == "__main__":
    main()
