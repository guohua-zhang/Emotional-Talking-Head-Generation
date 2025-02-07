import json
import warnings
import os
import pickle
import numpy as np
import torch
import librosa
from munch import Munch
from torch.utils import data


def load_audio(audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, _ = librosa.load(audio_path, sr=16000)
        return audio


class MEAD(data.Dataset):
    """Dataset class for the paired MEAD dataset."""

    emotions = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    levels = [3, 3, 3, 3, 1, 3, 3]
    emo_2_lvl = {e:l for e, l in zip(emotions, levels)}
    topk = 10

    def __init__(self, opt, which='source', phase='train', return_img_path=False):
        """Initialize the MEAD dataset."""
        super().__init__()
        self.cache_file = f'.cache/MEAD-dataset-{which}-{phase}.pkl'
        self.opt = opt
        self.which = which
        self.seq_len = opt.seq_len
        self.hop_len = opt.hop_len
        self.root = opt.train_root
        self.class_names = opt.class_names
        self.return_img_path = return_img_path
        with open(opt.dist_file, 'rb') as f:
            self.dists = pickle.load(f)
        self.dist_thresh = opt.dist_thresh
        if phase == 'train':
            if which == 'reference':
                self.selected_actors = opt.selected_actors_ref
                self.selected_emotions = opt.selected_emotions_ref
            else:
                self.selected_actors = opt.selected_actors
                #if opt.paired:
                #    assert opt.selected_emotions == ['neutral'], 'only support transfer from neutral to other emotion'
                self.selected_emotions = opt.selected_emotions
        elif phase == 'val':
            self.selected_actors = opt.selected_actors_val
            self.selected_emotions = opt.selected_emotions
        
        self.selected_labels = [self.class_names.index(
            e) for e in self.selected_emotions]

        self.seqs = []
        self.pseudos = []
        self.labels = []
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
            self.seqs = cache['seqs']
            self.pseudos = cache['pseudos']
            self.labels = cache['labels']
            print('load dataset from cache.')
            return True
        else:
            return False
    
    def load_data(self):
        if self.opt.paired and self.which != 'reference':
            self.prepare_pseudos()

        print('loading dataset...')
        for actor in self.selected_actors:
            actor_root = os.path.join(
                self.root, '{}_deca.pkl'.format(actor))
            assert os.path.isfile(
                actor_root), '%s is not a valid file' % actor_root

            actor_data = pickle.load(open(actor_root, "rb"))
            for emo in self.selected_emotions:
                index = 0
                for name in actor_data[emo]:
                    audio_path = os.path.join(self.root, actor, 'audio', emo, f'level_{self.emo_2_lvl[emo]}', name+'.m4a')
                    if not os.path.exists(audio_path):
                        continue
                    params = actor_data[emo][name]
                    params = np.concatenate((params[:, 0:1], params[:, 3:]), 1) # 51
                    for i in range(0, params.shape[0]-self.seq_len, self.hop_len):
                        frame_idx = i + self.seq_len - 1
                        img_path = os.path.join(self.root, actor, 'render', emo, f'level_{self.emo_2_lvl[emo]}', 'faces_aligned', name, f'{frame_idx:06d}.png')
                        self.seqs.append((params[i:i + self.seq_len], img_path))
                        self.labels.append(self.class_names.index(emo))
                        if not self.opt.paired or self.which == 'reference':
                            continue

                        self.pseudos.append([None] * len(self.class_names))
                        for label, emo_ in enumerate(self.class_names):
                            if emo_ == 'neutral':
                                continue
                            elif emo_ == emo:
                                self.pseudos[-1][label] = [self.seqs[-1]]
                                continue

                            dist = self.dists[actor][emo_][index]
                            inds = np.arange(dist.shape[0])[dist <= self.dist_thresh]
                            inds_ = np.argsort(dist[inds])
                            self.pseudos[-1][label] = [self._seqs[actor][emo_][j] for j in inds[inds_]]
                            if len(self.pseudos[-1][label]) > self.topk:
                                self.pseudos[-1][label] = self.pseudos[-1][label][:self.topk]

                        index += 1

        s = json.dumps({
            'root': self.root,
            'class_names': self.class_names,
            'actors': self.selected_actors,
            'emotions': self.selected_emotions,
            'seq_len': self.seq_len, 
            'hop_len': self.hop_len
        })
        cache_data = {
            'check_field': s, 
            'seqs': self.seqs, 
            'pseudos': self.pseudos, 
            'labels': self.labels}
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print('done.')
    
    def __getitem__(self, index):
        """Return one sequence and its corresponding label."""
        param, img_path = self.seqs[index]
        param = torch.FloatTensor(param)
        label = self.labels[index]
        if self.return_img_path:
            return index, param, label, img_path
        else:
            return index, param, label

    def __len__(self):
        """Return the number of sequences."""
        return len(self.seqs)
    
    def prepare_pseudos(self):
        print('preparing pseudos...')

        self._seqs = {}

        for actor in self.selected_actors:
            actor_root = os.path.join(
                self.root, '{}_deca.pkl'.format(actor))
            assert os.path.isfile(
                actor_root), '%s is not a valid file' % actor_root

            actor_data = pickle.load(open(actor_root, "rb"))

            self._seqs[actor] = {}
            for emo in self.class_names:
                if emo == 'neutral':
                    continue
                self._seqs[actor][emo] = []
                for name in actor_data[emo]:
                    audio_path = os.path.join(self.root, actor, 'audio', emo, f'level_{self.emo_2_lvl[emo]}', name+'.m4a')
                    if not os.path.exists(audio_path):
                        continue
                    params = actor_data[emo][name]
                    params = np.concatenate((params[:, 0:1], params[:, 3:]), 1) # 51
                    for i in range(0, params.shape[0]-self.seq_len, self.hop_len):
                        frame_idx = i + self.seq_len - 1
                        img_path = os.path.join(self.root, actor, 'render', emo, f'level_{self.emo_2_lvl[emo]}', 'faces_aligned', name, f'{frame_idx:06d}.png')
                        self._seqs[actor][emo].append((params[i:i + self.seq_len], img_path))

    def get_pseudo(self, index, label):
        if not self.opt.paired:
            return None
        pseudo_list = self.pseudos[index][label]
        if len(pseudo_list) == 0:
            return None
        else:
            choice = np.random.randint(0, len(pseudo_list), 1)
            param, img_path = pseudo_list[int(choice)]
            if self.return_img_path:
                return torch.FloatTensor(param), img_path
            else:
                return torch.FloatTensor(param)


def get_train_loader(opt, which):
    dataset = MEAD(opt, which)
    return data.DataLoader(dataset=dataset,
                           batch_size=opt.batch_size,
                           shuffle=True,
                           num_workers=opt.nThreads,
                           drop_last=True,
                           pin_memory=True)


def get_val_loader(opt, which):
    dataset = MEAD(opt, which, phase='val')
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

        self.iter = iter(self.loader)
        self.iter_ref = iter(self.loader_ref)

    def _fetch_inputs(self, label):
        try:
            ids, x, y = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            ids, x, y = next(self.iter)
        x_gt = []
        x_gt_mask = []
        for i, l in zip(ids, label):
            pseudo = self.loader.dataset.get_pseudo(i, int(l))
            if pseudo is None:
                pseudo = np.zeros_like(x[0])
                x_gt_mask.append(0)
            else:
                x_gt_mask.append(1)
            x_gt.append(pseudo)
        x_gt = torch.from_numpy(np.stack(x_gt, axis=0))
        x_gt_mask = torch.from_numpy(np.array(x_gt_mask, dtype=np.float32))
        return x, y, x_gt, x_gt_mask

    def _fetch_refs(self):
        try:
            _, x, y = next(self.iter_ref)
        except StopIteration:
            self.iter_ref = iter(self.loader_ref)
            _, x, y = next(self.iter_ref)
        return x, y

    def __next__(self):
        x_ref, y_ref = self._fetch_refs()
        x, y, x_gt, x_gt_mask = self._fetch_inputs(y_ref)
        z_trg = torch.randn(x.size(0), self.latent_dim)
        inputs = Munch(x_src=x, y_src=y, x_ref=x_ref,
                       y_ref=y_ref, z_trg=z_trg, 
                       x_gt=x_gt, x_gt_mask=x_gt_mask)

        return inputs


if __name__ == '__main__':
    opt = Munch(
        class_names=['neutral', 'happy'],
        train_root='MEAD-sim',
        seq_len=10,
        hop_len=1,
        selected_actors=['M003'],
        selected_emotions=['neutral']
    )

    dataset = MEAD(opt, dist_file='exp/similarity/dists/dists.pkl', dist_thresh=1.0)
    index, param, label = dataset[0]
    print(len(dataset))
    print(label, param.shape)
    print(dataset.get_pseudo(index, 1).shape)
    pseudos = dataset.pseudos
    lens = []
    for i in range(len(pseudos)):
        for x in pseudos[i]:
            if x is not None:
                lens.append(len(x))
    print(f'avg len(pseudos) = {sum(lens)/len(lens)}')
