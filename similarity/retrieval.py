import argparse
import torch
import os
import json
import librosa
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from models import ImageEncoder, AudioEncoder


class AlignedDataset(Dataset):
    _emotions = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    _levels = [3, 3, 3, 3, 1, 3, 3]
    _emo_2_lvl = {e:l for e, l in zip(_emotions, _levels)}
    
    def __init__(self, args, max_sample=2000):
        super().__init__()
        self.data_root = args.data_root
        with open(args.aligned_path) as f:
            self.aligned_path = json.load(f)
        self.actors = args.actors
        self.emotions = args.emotions
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
                        k = np.clip(i, 0, len(path1)-1)
                        fid_ = path1[k]
                        dst_clip = dst_audio[fid_*frame_len:(fid_+1)*frame_len]
                        if dst_clip.shape[0] == frame_len:
                            audio_batch.append(dst_clip)
                        if len(audio_batch) != 2:
                            continue

                        img_batch = [os.path.join(src_dir, f'{fid:06d}.png')]
                        k = np.clip(i, 0, len(path1)-1)
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
        
        if len(self.img_paths) > max_sample:
            inds = np.random.randint(0, len(self.img_paths), max_sample)
            self.img_paths = [self.img_paths[i] for i in inds]
            self.audio_clips = [self.audio_clips[i] for i in inds]
    
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


def distance(x, y):
    return ((x - y)**2).sum().sqrt()

@torch.no_grad()
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image_encoder = ImageEncoder(args.depth1, args.dim).to(device)
    audio_encoder = AudioEncoder(args.depth2, args.dim).to(device)
    ckpt = torch.load(args.ckpt, device)['model']
    image_encoder.load_state_dict(ckpt['image_encoder'])
    audio_encoder.load_state_dict(ckpt['audio_encoder'])
    image_encoder.eval()
    audio_encoder.eval()
    dataset = AlignedDataset(args)
    vis_idxs = np.random.randint(0, len(dataset), args.vis_num)
    

    for i, idx in enumerate(vis_idxs):
        src_image, src_audio = dataset[idx]
        src_image_vec = image_encoder(torch.from_numpy(src_image[:1]).to(device))
        src_audio_vec = audio_encoder(torch.from_numpy(src_audio[:1]).to(device))
        image_dists, audio_dists = [], []
        for j in tqdm(range(len(dataset)), desc=f'{idx} [{i+1}/{args.vis_num}]'):
            tgt_image, tgt_audio = dataset[j]
            tgt_image_vec = image_encoder(torch.from_numpy(tgt_image[1:]).to(device))
            tgt_audio_vec = audio_encoder(torch.from_numpy(tgt_audio[1:]).to(device))
            image_dists.append(distance(src_image_vec, tgt_image_vec))
            audio_dists.append(distance(src_audio_vec, tgt_audio_vec))
        print(torch.tensor(image_dists).mean(), torch.tensor(audio_dists).mean())
        dists, inds = torch.topk(torch.tensor(audio_dists), k=args.topk, largest=False)
        print(dists, inds)
        vis_dir = os.path.join(args.vis_dir, str(idx))
        os.makedirs(vis_dir, exist_ok=True)
        shutil.copy(dataset.img_paths[idx][0], os.path.join(vis_dir, f'src_{idx}.png'))
        for k in inds:
            shutil.copy(dataset.img_paths[k][1], os.path.join(vis_dir, f'tgt_{k}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth1',type=int,default=18, help='image encoder depth')
    parser.add_argument('--depth2',type=int,default=14, help='audio encoder depth')
    parser.add_argument('--dim',type=int,default=128)
    parser.add_argument('--ckpt',type=str,default='exp/similarity/finetune/last.pth', help='finetune checkpoint path')
    parser.add_argument('--data_root',type=str,default='MEAD-sim', help='data root')
    parser.add_argument('--aligned_path',type=str,default='MEAD-sim/aligned_path.json', help='aligned_path.json')
    parser.add_argument('--actors', type=str, nargs='+', help='Subset of the MEAD actors', default=['M003'])
    parser.add_argument('--emotions', type=str, nargs='+', help='Selection of Emotions', default=['neutral', 'happy'])
    parser.add_argument('--imsize',type=int,default=64, help='image size')
    parser.add_argument('--fps',type=int,default=30, help='video fps')
    parser.add_argument('--topk',type=int,default=3, help='retrieval topk')
    parser.add_argument('--vis_num',type=int,default=5, help='vis_num')
    parser.add_argument('--vis_dir',type=str,default='exp/similarity/vis')
    args = parser.parse_args()

    main()
