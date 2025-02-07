import argparse
import os
import cv2
import json
import math
import time
import torch
import numpy as np
from torch import nn
from torchvision.models import resnet
from torch.utils.data import Dataset, DataLoader
from models import ImageEncoder, Loss


class AlignedDataset(Dataset):
    _emotions = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    _levels = [3, 3, 3, 3, 1, 3, 3]
    _emo_2_lvl = {e:l for e, l in zip(_emotions, _levels)}
    
    def __init__(self, args):
        super().__init__()
        self.data_root = args.data_root
        with open(args.aligned_path) as f:
            self.aligned_path = json.load(f)
        self.actors = args.actors
        self.emotions = args.emotions
        self.nframe = args.nframe
        self.imsize = args.imsize
        self.img_paths = []

        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

        for actor in self.actors:
            for emo in self.emotions:
                if emo == 'neutral':
                    continue
                for k, v in self.aligned_path[actor][emo].items():
                    src, dst = k.split('_')
                    path0, path1 = v
                    src_dir = os.path.join(self.data_root, actor, 'image/front/neutral/level_1', src)
                    dst_dir = os.path.join(self.data_root, actor, 'image/front/', emo, f'level_{self._emo_2_lvl[emo]}', dst)
                    for i, fid in enumerate(path0):
                        batch = [os.path.join(src_dir, f'{fid:06d}.png')]
                        for j in range(math.ceil(-self.nframe/2), math.ceil(self.nframe/2)):
                            k = np.clip(i+j, 0, len(path1)-1)
                            batch.append(os.path.join(dst_dir, f'{path1[k]:06d}.png'))
                        all_exits = True
                        for x in batch:
                            if not os.path.exists(x):
                                # print(f'warning: image not exists {x}')
                                all_exits = False
                                break
                        if all_exits:
                            self.img_paths.append(batch)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        batch = self.img_paths[index]
        batch_data = []
        for x in batch:
            img = cv2.imread(x)[:, :, ::-1]
            img = cv2.resize(img, (self.imsize, self.imsize), interpolation=cv2.INTER_LINEAR)
            batch_data.append(img)
        batch_data = np.stack(batch_data, axis=0).transpose((0, 3, 1, 2))
        batch_data = (batch_data/255. - self.mean) / self.std
        return batch_data.astype(np.float32)


def distance(out):
    # same class
    dist0 = []
    for i in range(out.shape[1]):
        for j in range(out.shape[1]):
            if j == i:
                continue
            dist0.append(((out[:, i]-out[:, j])**2).sum(dim=1).sqrt())
    dist0 = torch.cat(dist0, dim=0).mean()

    # different class
    dist1 = []
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if j == i:
                continue
            dist1.append(((out[i]-out[j])**2).sum(dim=1).sqrt())
    dist1 = torch.cat(dist1, dim=0).mean()

    return dist0, dist1


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.work_dir, exist_ok=True)
    model = ImageEncoder(args.depth, args.dim).to(device)
    loss_fn = Loss(args.loss_mode, device)
    dataset = AlignedDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)

    for epoch in range(args.epochs):
        for step, imgs in enumerate(dataloader):
            bs, nframe, c, h, w = imgs.shape
            imgs = imgs.view(bs*nframe, c, h, w).to(device)
            out = model(imgs)
            out = out.view(bs, nframe, out.shape[-1])
            loss = loss_fn(out)
            loss.backward()
            optimizer.step()

            dist = distance(out)
            model.update_mdist(dist)

            if (step + 1) % args.log_interval == 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                      f'epoch {epoch+1}, step [{step+1}/{len(dataloader)}], loss {loss.item():.4f}, '
                      f'dist ({model.mdist[0]:.4f}, {model.mdist[1]:.4f}), lr {optimizer.param_groups[0]["lr"]:.4g}')

        scheduler.step()
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(ckpt, os.path.join(args.work_dir, 'last.pth'))
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth',type=int,default=18, help='model depth')
    parser.add_argument('--dim',type=int,default=128)
    parser.add_argument('--data_root',type=str,default='MEAD-sim', help='data root')
    parser.add_argument('--aligned_path',type=str,default='MEAD-sim/aligned_path.json', help='aligned_path.json')
    parser.add_argument('--actors', type=str, nargs='+', help='Subset of the MEAD actors', default=['M003'])
    parser.add_argument('--emotions', type=str, nargs='+', help='Selection of Emotions', default=['neutral', 'happy'])
    parser.add_argument('--imsize',type=int,default=64, help='image size')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--bs',type=int,default=8)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--wd',type=float,default=0., help='weight decay')
    parser.add_argument('--loss_mode', type=str, default='softmax')
    parser.add_argument('--nframe',type=int,default=1, help='num frames for positive samples')
    parser.add_argument('--work_dir',type=str,default='exp/similarity/image')
    parser.add_argument('--log_interval',type=int,default=50)
    args = parser.parse_args()

    main()
