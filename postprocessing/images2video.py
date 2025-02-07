import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from moviepy.editor import *


def main():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_path', type=str, default='.',
                        help="path to saved images")
    parser.add_argument('--out_path', type=str, default='.',
                        help="path to save video")
    parser.add_argument('--fps', type=float, default=30,
                        help=".")
    parser.add_argument('--audio', type=str, default=None,
                        help="Path to original .mp4 file that contains audio")

    args = parser.parse_args()
    if not args.imgs_path.endswith('/'):
        args.imgs_path += '/'
    img_paths = sorted(
        glob(os.path.join(args.imgs_path, '**/*.png'), recursive=True))
    dirs = set([os.path.dirname(p) for p in img_paths])
    h, w, _ = cv2.imread(img_paths[0]).shape

    for i, dir_ in enumerate(dirs):
        video_path = os.path.join(
            args.out_path, dir_.replace(args.imgs_path, '')) + '.mp4'
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        video = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
        for img in tqdm(sorted(filter(lambda x: dir_ in x, img_paths)),
                        desc=f'[{i+1}/{len(dirs)}] {os.path.basename(video_path)}'):
            video.write(cv2.imread(img))
        video.release()

        if args.audio is not None:
            print('Adding audio with MoviePy ...')
            video = VideoFileClip(video_path)
            video_audio = VideoFileClip(args.audio)
            video = video.set_audio(video_audio.audio)
            os.remove(video_path)
            video.write_videofile(video_path)

    print('DONE')


if __name__ == "__main__":
    main()
