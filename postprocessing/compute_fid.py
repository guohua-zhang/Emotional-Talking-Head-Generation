import argparse
import os
import torch
from glob import glob
from metrics.fid import compute_fid

def main(args):
    real_imgs = glob(os.path.join(args.real_img_dir, '**/*.png'), recursive=True)
    gen_imgs = glob(os.path.join(args.gen_img_dir, '**/*.png'), recursive=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fid = compute_fid(real_imgs, gen_imgs, device=device)
    print('fid =', fid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_img_dir', type=str, default='MEAD-sim/M003/render/happy/level_3/faces_aligned')
    parser.add_argument('--gen_img_dir', type=str, default='MEAD-sim/M003/render/neutral/level_1/reference_on_M009_happy_simi0/faces_aligned')
    args = parser.parse_args()

    main(args)
