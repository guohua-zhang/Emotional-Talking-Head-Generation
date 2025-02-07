import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def main():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--celeb', type=str, default='MEAD-sim/M003/render/neutral/level_1/', help="celeb path")
    parser.add_argument('--reference', type=str, default='celebrities/reference/M009_happy/images/000004.png', help="reference image")
    parser.add_argument('--exp_name', type=str, default='reference_on_M009_happy_simi0', help="exp name")
    parser.add_argument('--sub_dir', type=str, default='001', help="video name")
    parser.add_argument('--audio', type=str, default=None, help="Path to original .mp4 file that contains audio")
    parser.add_argument('--out_path', type=str, default='celebrities/out_videos/vis/simi0.mp4', help="path to save video")
    args = parser.parse_args()

    im_ref = cv2.imread(args.reference)
    h, w, _ = im_ref.shape

    video = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w*4, h))
    src_dir = os.path.join(args.celeb, 'images', args.sub_dir)
    tgt_dir = os.path.join(args.celeb, args.exp_name, 'images', args.sub_dir)
    shp_dir = os.path.join(args.celeb, args.exp_name, 'shapes', args.sub_dir)
    print('Converting images to video ...')
    for name in tqdm(sorted(os.listdir(tgt_dir))):
        im_src = cv2.imread(os.path.join(src_dir, name))
        im_shp = cv2.imread(os.path.join(shp_dir, name))
        im_tgt = cv2.imread(os.path.join(tgt_dir, name))
        im = np.zeros((h, w*4, 3), dtype=np.uint8)
        im[:, :w] = im_ref
        im[:, w:w*2] = im_src
        im[:, w*2:w*3] = im_shp
        im[:, w*3:w*4] = im_tgt
        video.write(im)

    cv2.destroyAllWindows()
    video.release()

    if args.audio is not None:
        from moviepy.editor import VideoFileClip
        print('Adding audio with MoviePy ...')
        video = VideoFileClip(args.out_path)
        video_audio = VideoFileClip(args.audio)
        video = video.set_audio(video_audio.audio)
        os.remove(args.out_path)
        video.write_videofile(args.out_path)

    print('DONE')


if __name__ == '__main__':
    main()
