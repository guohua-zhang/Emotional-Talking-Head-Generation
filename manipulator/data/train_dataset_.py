import json
import os
import pickle

import numpy as np
import torch
from munch import Munch
from torch.utils import data

from manipulator.data.util import align_audio


class MEAD(data.Dataset):
    """Dataset class for the MEAD dataset."""

    def __init__(self, opt, which='source', phase='train'):
        """Initialize the MEAD dataset."""

        self.which = which
        self.seq_len = opt.seq_len
        self.hop_len = opt.hop_len
        self.root = opt.train_root
        if phase == 'train':
            self.selected_actors = opt.selected_actors
        elif phase == 'val':
            self.selected_actors = opt.selected_actors_val

        self.selected_emotions = opt.selected_emotions

        self.seqs = []
        self.labels = []

        videos = []
        for actor in self.selected_actors:
            actor_root = os.path.join(self.root, '{}_deca.pkl'.format(actor))
            assert os.path.isfile(
                actor_root), '%s is not a valid file' % actor_root

            data_actor = pickle.load(open(actor_root, "rb"))
            videos.extend(data_actor)

        for v in videos:
            params, emotion = v
            params = np.concatenate((params[:, 0:1], params[:, 3:]), 1)
            seqs = [params[x:x + self.seq_len] for x in
                    range(0, params.shape[0], self.hop_len) if
                    len(params[x:x + self.seq_len]) == self.seq_len]
            f = False
            for i, e in enumerate(self.selected_emotions):
                if e == emotion:
                    label = i
                    f = True
            if not f:
                print(emotion)
                # raise
            else:
                self.seqs.extend(seqs)
                self.labels.extend([label]*len(seqs))
        self.seqs = np.stack(self.seqs, axis=0)

        self.num_seqs = len(self.seqs)

        if self.which == 'reference':
            p = np.random.permutation(self.num_seqs)

            self.seqs = self.seqs[p]
            self.labels = np.array(self.labels)[p].tolist()

    def __getitem__(self, index):
        """Return one sequence and its corresponding label."""

        sequence = torch.FloatTensor(self.seqs[index])
        label = self.labels[index]

        return sequence, label

    def __len__(self):
        """Return the number of sequences."""
        return len(self.seqs)


class PairedMEAD(data.Dataset):
    """Dataset class for the paired MEAD dataset."""

    emotions = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    levels = [3, 3, 3, 3, 1, 3, 3]
    emo_2_lvl = {e:l for e, l in zip(emotions, levels)}

    def __init__(self, opt, which='source', phase='train', paired_info='paired_info.json'):
        """Initialize the MEAD dataset."""
        self.cache_file = f'.cache/paired_seqs-{which}-{phase}.pkl'
        self.which = which
        self.seq_len = opt.seq_len
        self.hop_len = opt.hop_len
        self.root = opt.train_root
        self.class_names = opt.class_names
        if phase == 'train':
            if which == 'reference':
                self.selected_actors = opt.selected_actors_ref
                self.selected_emotions = opt.selected_emotions_ref
            else:
                self.selected_actors = opt.selected_actors
                self.selected_emotions = opt.selected_emotions
        elif phase == 'val':
            self.selected_actors = opt.selected_actors_val
            self.selected_emotions = opt.selected_emotions
        
        assert os.path.exists(os.path.join(self.root, paired_info))
        with open(os.path.join(self.root, paired_info)) as f:
            self.paired_info = json.load(f)

        self.selected_labels = [self.class_names.index(
            e) for e in self.selected_emotions]

        self.seqs = []
        if not self.check_cache():
            self.load_data()

        self.num_seqs = len(self.seqs)

        if self.which == 'reference':
            np.random.shuffle(self.seqs)
    
    def check_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        if not os.path.exists(self.cache_file):
            return False
        with open(self.cache_file,'rb') as f:
            cache = pickle.load(f)
        check_field = cache['check_field']
        s = json.dumps({
            'root': self.root,
            'class_names': self.class_names,
            'actors': self.selected_actors,
            'emotions': self.selected_emotions,
            'seq_len': self.seq_len, 
            'hop_len': self.hop_len
        })
        if check_field == s:
            self.seqs = cache['data']
            print('load dataset from cache.')
            return True
        else:
            return False
    
    def align_sequences(self, seqs, data_dir):
        aligned_seqs = []
        for seq in seqs:
            longest, m, i = seq[0], seq[0][1].shape[0], 0
            for j, item in enumerate(seq):
                if item[1].shape[0] > m:
                    longest = item
                    m = item[1].shape[0]
                    i = j
            audio_i = os.path.join(data_dir, 'audio', self.class_names[i], f'level_{self.emo_2_lvl[self.class_names[i]]}', longest[0]+'.m4a')
            res = [None for _ in range(len(self.class_names))]
            for j, (name, params) in enumerate(seq):
                n = params.shape[0]
                if n != m:
                    audio_j = os.path.join(data_dir, 'audio', self.class_names[j], f'level_{self.emo_2_lvl[self.class_names[j]]}', name+'.m4a')
                    path = align_audio(audio_i, audio_j)
                    params = np.stack([params[min(k, n-1), :] for k in path[1]][:m], axis=0)
                
                res[j] = [params[x:x + self.seq_len] for x in range(0, params.shape[0]-self.seq_len, self.hop_len)]

            aligned_seqs.extend(zip(*res))
        return aligned_seqs
    
    def load_data(self):
        print('loading dataset...')
        for actor in self.selected_actors:
            actor_root = os.path.join(
                self.root, '{}_deca.pkl'.format(actor))
            assert os.path.isfile(
                actor_root), '%s is not a valid file' % actor_root

            actor_data = pickle.load(open(actor_root, "rb"))
            _paired_info = self.paired_info[actor]
            num_videos = len(_paired_info[self.class_names[0]])
            seqs = []
            for i in range(num_videos):
                line = []
                broken = False
                for emotion in self.class_names:
                    video = _paired_info[emotion][i]
                    if video not in actor_data[emotion]:
                        broken = True
                        break
                    params = actor_data[emotion][video]
                    params = np.concatenate((params[:, 0:1], params[:, 3:]), 1) # 51
                    line.append((video, params))
                if not broken:
                    seqs.append(line)
            if len(seqs) > 0:
                self.seqs.extend(self.align_sequences(seqs, os.path.join(self.root, actor)))

        s = json.dumps({
            'root': self.root,
            'class_names': self.class_names,
            'actors': self.selected_actors,
            'emotions': self.selected_emotions,
            'seq_len': self.seq_len, 
            'hop_len': self.hop_len
        })
        cache_data = {'check_field': s, 'data': self.seqs}
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print('done.')
    
    def __getitem__(self, index):
        """Return one sequence and its corresponding label."""

        label = np.random.choice(self.selected_labels)
        sequence = torch.FloatTensor(self.seqs[index][label])

        return index, sequence, label

    def __len__(self):
        """Return the number of sequences."""
        return len(self.seqs)

    def get_paired_gt(self, index, label):
        return torch.FloatTensor(self.seqs[index][label])


def get_train_loader(opt, which):
    dataset = PairedMEAD(opt, which) if opt.paired else MEAD(opt, which)
    return data.DataLoader(dataset=dataset,
                           batch_size=opt.batch_size,
                           shuffle=True,
                           num_workers=opt.nThreads,
                           drop_last=True,
                           pin_memory=True)


def get_val_loader(opt, which):
    dataset = PairedMEAD(opt, which, phase='val') if opt.paired else MEAD(
        opt, which, phase='val')
    return data.DataLoader(dataset=dataset,
                           batch_size=opt.batch_size,
                           shuffle=False,
                           num_workers=opt.nThreads,
                           drop_last=True,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref, latent_dim=4):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, y = next(self.iter_ref)
        return x, y

    def __next__(self):
        x, y = self._fetch_inputs()
        x_ref, y_ref = self._fetch_refs()
        z_trg = torch.randn(x.size(0), self.latent_dim)
        inputs = Munch(x_src=x, y_src=y, x_ref=x_ref, y_ref=y_ref, z_trg=z_trg)

        return inputs


class PairedInputFetcher:
    def __init__(self, loader, loader_ref, latent_dim=4):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim

        self.iter = iter(self.loader)
        self.iter_ref = iter(self.loader_ref)

    def _fetch_inputs(self, label):
        try:
            ids, x, y = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            ids, x, y = next(self.iter)
        x_gt = []
        for i, l in zip(ids, label):
            x_gt.append(self.loader.dataset.get_paired_gt(i, int(l)))
        x_gt = torch.from_numpy(np.stack(x_gt, axis=0))
        return x, y, x_gt

    def _fetch_refs(self):
        try:
            _, x, y = next(self.iter_ref)
        except StopIteration:
            self.iter_ref = iter(self.loader_ref)
            _, x, y = next(self.iter_ref)
        return x, y

    def __next__(self):
        x_ref, y_ref = self._fetch_refs()
        x, y, x_gt = self._fetch_inputs(y_ref)
        z_trg = torch.randn(x.size(0), self.latent_dim)
        inputs = Munch(x_src=x, y_src=y, x_ref=x_ref,
                       y_ref=y_ref, z_trg=z_trg, x_gt=x_gt)

        return inputs


if __name__ == '__main__':
    opt = Munch(
        class_names=['neutral', 'happy'],
        train_root='MEAD-sim',
        seq_len=10,
        hop_len=1,
        selected_actors=['W019', 'W016', 'M023', 'M034', 'M037', 'W037', 'M027'],
        selected_emotions=['neutral']
    )

    dataset = PairedMEAD(opt)
    index, sequence, label = dataset[0]
    print(len(dataset))
    print(label, sequence.shape)
    print(dataset.get_paired_gt(index, 1).shape)
