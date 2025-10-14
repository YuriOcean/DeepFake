import os
import cv2
import time
import shutil
import random
import datetime
import argparse
import numpy as np
import logging as logger
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch_kmeans import KMeans
from torch_kmeans.utils.distances import CosineSimilarity
from sklearn.metrics import f1_score
import csv
import ast
from PIL import Image
from pycocotools import mask as mask_util

from losses import MyInfoNCE
from models.vit import FOCAL_ViT
from models.hrnet import FOCAL_HRNet

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s] %(message)s',
                   datefmt='%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='train', help='one of [train, val, test_single]')
parser.add_argument('--input_size', type=int, default=1024, help='size of resized input')
parser.add_argument('--gt_ratio', type=int, default=16, help='resolution of input / output, 4 for HRNet and 16 for ViT')
parser.add_argument('--train_bs', type=int, default=1, help='training batch size')
parser.add_argument('--test_bs', type=int, default=1, help='testing batch size')
parser.add_argument('--save_res', type=int, default=1, help='whether to save the output')
parser.add_argument('--gpu', type=str, default='0,1', help='GPU ID')
parser.add_argument('--metric', type=str, default='cosine', help='metric for loss and clustering')
parser.add_argument('--train_csv', type=str, default='/data/disk2/yer/Dataset/train/train_split.csv')
parser.add_argument('--test_csv', type=str, default='/data/disk2/yer/Dataset/train/test_split.csv')
parser.add_argument('--root_dir', type=str, default='/data/disk2/yer/Dataset/train/')
args = parser.parse_args()
logger.info(args)

date_now = datetime.datetime.now()
date_now = 'Log_v%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
args.out_dir = date_now

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(666666)
torch.manual_seed(666666)
torch.cuda.manual_seed(666666)
torch.backends.cudnn.deterministic = True


# ---------------------- Dataset ----------------------
class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir, choice='train', input_size=1024, gt_ratio=16):
        self.root_dir = root_dir
        self.choice = choice
        self.input_size = input_size
        self.gt_ratio = gt_ratio

        self.samples = []
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    'path': row['Path'],
                    'rle': row['Region']
                })

        self.albu = A.Compose([
            A.RandomScale(scale_limit=(-0.5, 0.0), p=0.75),
            A.PadIfNeeded(min_height=self.input_size, min_width=self.input_size, p=1.0),
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
                A.Transpose(p=1),
            ], p=0.75),
            A.ImageCompression(quality_lower=50, quality_upper=95, p=0.75),
        ])

        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def rle_to_mask(self, rle_input):
        try:
            if isinstance(rle_input, str) and rle_input.strip() != "":
                rle_dict = ast.literal_eval(rle_input)
                if isinstance(rle_dict['counts'], str):
                    rle_dict['counts'] = rle_dict['counts'].encode('utf-8')
                mask = mask_util.decode(rle_dict)
                return mask.astype(np.uint8)
            else:
                return None
        except Exception as e:
            print(f"RLE转Mask失败: {e}")
            return None

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.root_dir, sample['path'])
        img = np.array(Image.open(img_path).convert('RGB'))

        mask = self.rle_to_mask(sample['rle'])
        if mask is None:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            mask = np.array(Image.fromarray(mask).resize(
                (img.shape[1], img.shape[0]), resample=Image.NEAREST
            ))

        H, W, _ = img.shape

        if self.choice == 'train' and random.random() < 0.75:
            augmented = self.albu(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = cv2.resize(img, (self.input_size, self.input_size))
        mask = cv2.resize(mask, (self.input_size // self.gt_ratio, self.input_size // self.gt_ratio), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.
        mask = mask.astype(np.float32) / 255.

        img_tensor = self.transform(img)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return img_tensor, mask_tensor, H, W, os.path.basename(sample['path'])


# ---------------------- FOCAL Wrapper ----------------------
class FOCAL(nn.Module):
    def __init__(self, net_list=[('ViT', '')]):
        super(FOCAL, self).__init__()
        self.lr = 1e-4
        self.network_list = []
        for net_name, net_weight in net_list:
            if net_name == 'HRNet':
                cur_net = FOCAL_HRNet()
            elif net_name == 'ViT':
                cur_net = FOCAL_ViT()
            else:
                logger.info('Error: Undefined Network.')
                exit()
            cur_net = nn.DataParallel(cur_net).cuda()
            if net_weight != '':
                self.load(cur_net, net_weight)
            self.network_list.append(cur_net)

        self.extractor_optimizer = optim.Adam(self.network_list[0].parameters(), lr=self.lr)
        self.save_dir = 'weights/' + args.out_dir
        if args.type == 'train':
            rm_and_make_dir(self.save_dir)
        self.myInfoNCE = MyInfoNCE(metric=args.metric)
        self.clustering = KMeans(verbose=False, n_clusters=2, distance=CosineSimilarity)

    def process(self, Ii, Mg, isTrain=False):
        self.extractor_optimizer.zero_grad()

        if isTrain:
            Fo = self.network_list[0](Ii)
            Fo = Fo.permute(0, 2, 3, 1)
            B, H, W, C = Fo.shape
            Fo = F.normalize(Fo, dim=3)
        else:
            with torch.no_grad():
                Fo = self.network_list[0](Ii)
                Fo = Fo.permute(0, 2, 3, 1)
                B, H, W, C = Fo.shape
                Fo = F.normalize(Fo, dim=3)
                Fo_list = [Fo]
                for additional_net in self.network_list[1:]:
                    Fo_add = additional_net(Ii)
                    Fo_add = F.interpolate(Fo_add, (H, W))
                    Fo_add = Fo_add.permute(0, 2, 3, 1)
                    Fo_add = F.normalize(Fo_add, dim=3)
                    Fo_list.append(Fo_add)
                Fo = torch.cat(Fo_list, dim=3)

        if isTrain:
            info_nce_loss = []
            for idx in range(B):
                Fo_idx = Fo[idx]
                Mg_idx = Mg[idx][0]
                query = Fo_idx[Mg_idx == 0]
                negative = Fo_idx[Mg_idx == 1]
                if negative.size(0) == 0 or query.size(0) == 0:
                    continue
                dict_size = 1000
                query_sample = query[torch.randperm(query.size()[0])[:dict_size]]
                negative_sample = negative[torch.randperm(negative.size(0))[:dict_size]]
                info_nce_loss.append(self.myInfoNCE(query_sample, query_sample, negative_sample))

            batch_loss = torch.mean(torch.stack(info_nce_loss).squeeze())
            self.backward(batch_loss)
            return batch_loss
        else:
            Mo = None
            Fo = torch.flatten(Fo, start_dim=1, end_dim=2)
            result = self.clustering(x=Fo, k=2)
            Lo_batch = result.labels
            for idx in range(B):
                Lo = Lo_batch[idx]
                if torch.sum(Lo) > torch.sum(1 - Lo):
                    Lo = 1 - Lo
                Lo = Lo.view(H, W)[None, :, :, None]
                Mo = torch.cat([Mo, Lo], dim=0) if Mo is not None else Lo
            Mo = Mo.permute(0, 3, 1, 2)
            return Mo

    def backward(self, batch_loss=None):
        if batch_loss is not None:
            batch_loss.backward(retain_graph=False)
            self.extractor_optimizer.step()

    def save(self, path=''):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.network_list[0].state_dict(),
                   self.save_dir + path + '%s_weights.pth' % self.network_list[0].module.name)

    def load(self, extractor, path=''):
        weights_file = torch.load('weights/' + path)
        cur_weights = extractor.state_dict()
        for key in weights_file:
            if key in cur_weights.keys() and weights_file[key].shape == cur_weights[key].shape:
                cur_weights[key] = weights_file[key]
        extractor.load_state_dict(cur_weights)
        logger.info('Loaded [%s] from [%s]' % (extractor.module.name, path))


# ---------------------- Forgery Forensics ----------------------
class ForgeryForensics():
    def __init__(self):
        self.train_csv = args.train_csv
        self.test_csv = args.test_csv
        self.root_dir = args.root_dir

        # Dataset
        train_dataset = MyDataset(csv_file=self.train_csv, root_dir=self.root_dir, choice='train',
                                  input_size=args.input_size, gt_ratio=args.gt_ratio)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_bs, num_workers=args.train_bs, shuffle=True)
        logger.info('Train on %d images.' % len(train_dataset))

        self.test_dataset = MyDataset(csv_file=self.test_csv, root_dir=self.root_dir, choice='test',
                                      input_size=args.input_size, gt_ratio=args.gt_ratio)
        logger.info('Test on %d images.' % len(self.test_dataset))

        # FOCAL Model
        self.focal = FOCAL([
            ('ViT', ''),
        ]).cuda()

        self.n_epochs = 99

    def train(self):
        cnt, batch_losses = 0, []
        best_score = 0
        scheduler = ReduceLROnPlateau(self.focal.extractor_optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-8)
        self.focal.train()
        for epoch in range(1, self.n_epochs + 1):
            for items in self.train_loader:
                cnt += args.train_bs
                Ii, Mg = (item.cuda() for item in items[:2])
                batch_loss = self.focal.process(Ii, Mg, isTrain=True)
                batch_losses.append(batch_loss.item())
                if cnt % (args.train_bs * 20) == 0:
                    logger.info('Tra (%6d/%6d): G:%5.4f' % (cnt, len(self.train_loader.dataset), np.mean(batch_losses)))
                if cnt % int((len(self.train_loader.dataset) / 80) // args.train_bs * args.train_bs) == 0:
                    self.focal.save('latest/')
                    logger.info('Ep%03d(%6d/%6d): Tra: G:%5.4f' % (epoch, cnt, len(self.train_loader.dataset), np.mean(batch_losses)))
                    tmp_score = self.val()
                    scheduler.step(tmp_score)
                    if tmp_score > best_score:
                        best_score = tmp_score
                        logger.info('Score: %5.4f (Best)' % best_score)
                        self.focal.save('Ep%03d_%5.4f/' % (epoch, tmp_score))
                    else:
                        logger.info('Score: %5.4f' % tmp_score)
                    self.focal.train()
                    batch_losses = []
            cnt = 0

    def val(self):
        P_F1, P_IOU = ForensicTesting(self.focal, bs=args.test_bs, test_dataset=self.test_dataset)
        logger.info('Validation Score: PF1:%5.4f, PIOU:%5.4f' % (P_F1, P_IOU))
        return P_F1


def ForensicTesting(model, bs=1, test_dataset=None):
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=min(48, bs), shuffle=False)
    for net in model.network_list:
        net.eval()

    f1_list, iou_list = [], []

    for items in test_loader:
        Ii, Mg, Hg, Wg, filename = (item.cuda() for item in items)
        Mo = model.process(Ii, None, isTrain=False)
        Mg, Mo = convert(Mg), convert(Mo)

        for i in range(Mo.shape[0]):
            Mo_resized = thresholding(cv2.resize(Mo[i], (Mg[i].shape[:2][::-1])))[..., None]
            f1_list.append(f1_score(Mg[i].flatten(), Mo_resized.flatten(), average='macro'))
            iou_list.append(metric_iou(Mo_resized / 255., Mg[i] / 255.))

    return np.mean(f1_list), np.mean(iou_list)


def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def convert(x):
    x = x * 255.
    return x.permute(0, 2, 3, 1).cpu().detach().numpy()


def thresholding(x, thres=0.5):
    x[x <= int(thres * 255)] = 0
    x[x > int(thres * 255)] = 255
    return x


def metric_iou(prediction, groundtruth):
    intersection = np.logical_and(prediction, groundtruth)
    union = np.logical_or(prediction, groundtruth)
    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    if np.sum(intersection) + np.sum(union) == 0:
        iou = 1
    return iou


# ---------------------- Main ----------------------
if __name__ == '__main__':
    if args.type == 'train':
        model = ForgeryForensics()
        model.train()
    elif args.type == 'val':
        model = ForgeryForensics()
        model.val()
    elif args.type == 'test_single':
        model = ForgeryForensics()
        ForensicTesting(model.focal, bs=args.test_bs, test_dataset=model.test_dataset)
