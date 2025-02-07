"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import pickle
import numpy as np
import scipy.linalg
import torch
from tqdm import tqdm
from . import metric_utils

import sys
sys.path.append('/home/zihua/workspace/stylegan3')

def compute_fid(real_imgs, gen_imgs, batch_size=8, device=torch.device('cuda')):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    with metric_utils.open_url(detector_url, verbose=True) as f:
        detector = pickle.load(f).to(device)
    
    real_loader = metric_utils.get_dataloader(real_imgs, batch_size)
    gen_loader = metric_utils.get_dataloader(gen_imgs, batch_size)
    stats_real = metric_utils.FeatureStats(max_items=len(real_loader.dataset))
    stats_gen = metric_utils.FeatureStats(max_items=len(gen_loader.dataset))

    for images in tqdm(real_loader, desc='calculating real images'):
        features = detector(images.to(device), return_features=True)
        stats_real.append_torch(features)
    
    for images in tqdm(gen_loader, desc='calculating generated images'):
        features = detector(images.to(device), return_features=True)
        stats_gen.append_torch(features)

    print('getting mean cov...')
    mu_real, sigma_real = stats_real.get_mean_cov()
    mu_gen, sigma_gen = stats_gen.get_mean_cov()

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)
