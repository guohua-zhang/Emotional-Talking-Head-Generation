import argparse
import os
import librosa
import json
import math
import time
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from models import AudioEncoder, Loss


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
        self.fps = args.fps
        self.audio_clips = []

        for actor in self.actors:
            for emo in self.emotions:
                if emo == 'neutral':
                    continue
                for k, v in self.aligned_path[actor][emo].items():
                    src, dst = k.split('_')
                    path0, path1 = v
                    src_audio = os.path.join(self.data_root, actor, 'audio/neutral/level_1', src+'.m4a')
                    dst_audio = os.path.join(self.data_root, actor, 'audio', emo, f'level_{self._emo_2_lvl[emo]}', dst+'.m4a')
                    src_audio, _ = librosa.load(src_audio, sr=16000)
                    dst_audio, _ = librosa.load(dst_audio, sr=16000)
                    frame_len = 16000 // self.fps
                    for i, fid in enumerate(path0):
                        src_clip = src_audio[fid*frame_len:(fid+1)*frame_len]
                        if src_clip.shape[0] != frame_len:
                            continue
                        batch = [src_clip]
                        for j in range(math.ceil(-self.nframe/2), math.ceil(self.nframe/2)):
                            k = np.clip(i+j, 0, len(path1)-1)
                            fid_ = path1[k]
                            dst_clip = dst_audio[fid_*frame_len:(fid_+1)*frame_len]
                            if dst_clip.shape[0] == frame_len:
                                batch.append(dst_clip)
                        if len(batch) == self.nframe + 1:
                            self.audio_clips.append(batch)
    
    def __len__(self):
        return len(self.audio_clips)
    
    def __getitem__(self, index):
        batch = self.audio_clips[index]
        batch_data = np.stack(batch, axis=0)[:, np.newaxis, :]
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
    model = AudioEncoder(args.depth, args.dim).to(device)
    loss_fn = Loss(args.loss_mode, device)
    dataset = AlignedDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)

    for epoch in range(args.epochs):
        for step, audios in enumerate(dataloader):
            bs, nframe, c, l = audios.shape
            audios = audios.view(bs*nframe, c, l).to(device)
            out = model(audios)
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
    parser.add_argument('--depth',type=int,default=14, help='model depth')
    parser.add_argument('--dim',type=int,default=128)
    parser.add_argument('--data_root',type=str,default='MEAD-sim', help='data root')
    parser.add_argument('--aligned_path',type=str,default='MEAD-sim/aligned_path.json', help='aligned_path.json')
    parser.add_argument('--actors', type=str, nargs='+', help='Subset of the MEAD actors', default=['M003'])
    parser.add_argument('--emotions', type=str, nargs='+', help='Selection of Emotions', default=['neutral', 'happy'])
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--bs',type=int,default=8)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--wd',type=float,default=0., help='weight decay')
    parser.add_argument('--loss_mode', type=str, default='softmax')
    parser.add_argument('--nframe',type=int,default=1, help='num frames for positive samples')
    parser.add_argument('--fps',type=int,default=30, help='video fps')
    parser.add_argument('--work_dir',type=str,default='exp/similarity/audio')
    parser.add_argument('--log_interval',type=int,default=50)
    args = parser.parse_args()

    main()
