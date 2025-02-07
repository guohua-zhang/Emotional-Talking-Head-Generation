import argparse
import os
import cv2
import json
import math
import time
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

from models import ImageEncoder, AudioEncoder, Loss


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
        self.fps = args.fps
        self.img_paths = []
        self.audio_clips = []

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
                    src_audio = os.path.join(self.data_root, actor, 'audio/neutral/level_1', src+'.m4a')
                    dst_audio = os.path.join(self.data_root, actor, 'audio', emo, f'level_{self._emo_2_lvl[emo]}', dst+'.m4a')
                    src_audio, _ = librosa.load(src_audio, sr=16000)
                    dst_audio, _ = librosa.load(dst_audio, sr=16000)
                    frame_len = 16000 // self.fps
                    for i, fid in enumerate(path0):
                        src_clip = src_audio[fid*frame_len:(fid+1)*frame_len]
                        if src_clip.shape[0] != frame_len:
                            continue
                        audio_batch = [src_clip]
                        for j in range(math.ceil(-self.nframe/2), math.ceil(self.nframe/2)):
                            k = np.clip(i+j, 0, len(path1)-1)
                            fid_ = path1[k]
                            dst_clip = dst_audio[fid_*frame_len:(fid_+1)*frame_len]
                            if dst_clip.shape[0] == frame_len:
                                audio_batch.append(dst_clip)
                        if len(audio_batch) != self.nframe + 1:
                            continue

                        img_batch = [os.path.join(src_dir, f'{fid:06d}.png')]
                        for j in range(math.ceil(-self.nframe/2), math.ceil(self.nframe/2)):
                            k = np.clip(i+j, 0, len(path1)-1)
                            img_batch.append(os.path.join(dst_dir, f'{path1[k]:06d}.png'))
                        
                        all_exits = True
                        for x in img_batch:
                            if not os.path.exists(x):
                                # print(f'warning: image not exists {x}')
                                all_exits = False
                                break
                        if all_exits:
                            self.audio_clips.append(audio_batch)
                            self.img_paths.append(img_batch)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_batch = self.img_paths[index]
        audio_batch = self.audio_clips[index]
        img_batch_data = []
        for x in img_batch:
            img = cv2.imread(x)[:, :, ::-1]
            img = cv2.resize(img, (self.imsize, self.imsize), interpolation=cv2.INTER_LINEAR)
            img_batch_data.append(img)
        img_batch_data = np.stack(img_batch_data, axis=0).transpose((0, 3, 1, 2))
        img_batch_data = (img_batch_data/255. - self.mean) / self.std
        audio_batch_data = np.stack(audio_batch, axis=0)[:, np.newaxis, :]
        return img_batch_data.astype(np.float32), audio_batch_data.astype(np.float32)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.work_dir, exist_ok=True)
    image_encoder = ImageEncoder(args.depth1, args.dim).to(device)
    image_ckpt = torch.load(args.image_ckpt, device)['model']
    image_encoder.load_state_dict(image_ckpt)
    audio_encoder = AudioEncoder(args.depth2, args.dim).to(device)
    audio_ckpt = torch.load(args.audio_ckpt, device)['model']
    audio_encoder.load_state_dict(audio_ckpt)
    loss_fn = Loss(args.loss_mode, device)
    dataset = AlignedDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    optimizer = torch.optim.Adam(list(image_encoder.parameters())+list(audio_encoder.parameters()), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)

    for epoch in range(args.epochs):
        for step, (imgs, audios) in enumerate(dataloader):
            bs, nframe, c, h, w = imgs.shape
            imgs = imgs.view(bs*nframe, c, h, w).to(device)
            bs, nframe, c, l = audios.shape
            audios = audios.view(bs*nframe, c, l).to(device)
            image_vec = image_encoder(imgs)
            audio_vec = audio_encoder(audios)
            image_vec = image_vec.view(bs, nframe, image_vec.shape[-1])
            audio_vec = audio_vec.view(bs, nframe, audio_vec.shape[-1])
            loss_image, image_sims = loss_fn(image_vec)
            loss_audio, audio_sims = loss_fn(audio_vec)
            
            loss_consis = args.lambd * torch.abs(image_sims.softmax(dim=1) - audio_sims.softmax(dim=1)).mean()
            loss = loss_image + loss_audio + loss_consis
            loss.backward()
            optimizer.step()

            image_encoder.update_mdist(image_vec)
            audio_encoder.update_mdist(audio_vec)

            if (step + 1) % args.log_interval == 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                      f'epoch {epoch+1}, step [{step+1}/{len(dataloader)}], loss_image {loss_image.item():.4f}, loss_audio {loss_audio.item():.4f}, ' 
                      f'loss_consis {loss_consis.item():.4f}, image dist ({image_encoder.mdist[0].item():.4f}, {image_encoder.mdist[1].item():.4f}), '
                      f'audio dist ({audio_encoder.mdist[0].item():.4f}, {audio_encoder.mdist[1].item():.4f}) lr {optimizer.param_groups[0]["lr"]:.4g}')
        
        scheduler.step()
        
        ckpt = {
            'model': {
                'image_encoder': image_encoder.state_dict(),
                'audio_encoder': audio_encoder.state_dict()
            },
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(ckpt, os.path.join(args.work_dir, 'last.pth'))
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth1',type=int,default=18, help='image encoder depth')
    parser.add_argument('--depth2',type=int,default=14, help='audio encoder depth')
    parser.add_argument('--dim',type=int,default=128)
    parser.add_argument('--image_ckpt',type=str,default='exp/similarity/image/last.pth', help='image encoder checkpoint path')
    parser.add_argument('--audio_ckpt',type=str,default='exp/similarity/audio/last.pth', help='audio encoder checkpoint path')
    parser.add_argument('--data_root',type=str,default='MEAD-sim', help='data root')
    parser.add_argument('--aligned_path',type=str,default='MEAD-sim/aligned_path.json', help='aligned_path.json')
    parser.add_argument('--actors', type=str, nargs='+', help='Subset of the MEAD actors', default=['M003'])
    parser.add_argument('--emotions', type=str, nargs='+', help='Selection of Emotions', default=['neutral', 'happy'])
    parser.add_argument('--imsize',type=int,default=64, help='image size')
    parser.add_argument('--fps',type=int,default=30, help='video fps')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--bs',type=int,default=8)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--wd',type=float,default=0., help='weight decay')
    parser.add_argument('--loss_mode', type=str, default='softmax')
    parser.add_argument('--nframe',type=int,default=1, help='num frames for positive samples')
    parser.add_argument('--lambd',type=float,default=1.0, help='consistency loss weight')
    parser.add_argument('--work_dir',type=str,default='exp/similarity/finetune')
    parser.add_argument('--log_interval',type=int,default=50)
    args = parser.parse_args()

    main()
