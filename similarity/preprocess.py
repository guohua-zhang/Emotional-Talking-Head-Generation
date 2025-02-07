import os
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy import ndimage
from facenet_pytorch import MTCNN, extract_face
from align_audio import align_audio

emotions = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
levels = [3, 3, 3, 3, 1, 3, 3]
emo_2_lvl = {e:l for e, l in zip(emotions, levels)}
filter_length = 500
cropped_img_size = 256
_margin = 70
_window_length = 49
height_recentre = 0.


def tensor2npimage(image_tensor, imtype=np.uint8):
    # Tesnor in range [0,255]
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2npimage(image_tensor[i], imtype))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, transpose=True):
    if transpose:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_images(images, save_dir, start_i):
    for i in range(len(images)):
        n_frame = "{:06d}".format(i + start_i)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_image(images[i], os.path.join(save_dir, n_frame + '.png'))


def smooth_boxes(boxes, previous_box):
    # Check if there are None boxes.
    if boxes[0] is None:
        boxes[0] = previous_box
    for i in range(len(boxes)):
        if boxes[i] is None:
            boxes[i] = next((item for item in boxes[i+1:] if item is not None), boxes[i-1])
    boxes = [box[0] for box in boxes]   # if more than one faces detected, keep the one with the heighest probability
    # Smoothen boxes
    old_boxes = np.array(boxes)
    window_length = min(_window_length, old_boxes.shape[0])
    if window_length % 2 == 0:
        window_length -= 1
    smooth_boxes = np.concatenate([ndimage.median_filter(old_boxes[:,i], size=window_length, mode='reflect').reshape((-1,1)) for i in range(4)], 1)
    # Make boxes square.
    for i in range(len(smooth_boxes)):
        offset_w = smooth_boxes[i][2] - smooth_boxes[i][0]
        offset_h = smooth_boxes[i][3] - smooth_boxes[i][1]
        offset_dif = (offset_h - offset_w) / 2
        # width
        smooth_boxes[i][0] = smooth_boxes[i][2] - offset_w - offset_dif
        smooth_boxes[i][2] = smooth_boxes[i][2] + offset_dif
        # height - center a bit lower
        smooth_boxes[i][3] = smooth_boxes[i][3] + height_recentre * offset_h
        smooth_boxes[i][1] = smooth_boxes[i][3] - offset_h

    return smooth_boxes

def get_faces(detector, images, previous_box):
    ret_faces = []
    ret_boxes = []

    all_boxes = []
    all_imgs = []

    # Get bounding boxes
    for lb in np.arange(0, len(images), 8):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+8]]
        boxes, _ = detector.detect(imgs_pil)
        all_boxes.extend(boxes)
        all_imgs.extend(imgs_pil)
    # Temporal smoothing
    boxes = smooth_boxes(all_boxes, previous_box)
    # Crop face regions.
    for img, box in zip(all_imgs, boxes):
        face = extract_face(img, box, cropped_img_size, _margin)
        ret_faces.append(face)
        # Find real bbox   (taken from https://github.com/timesler/facenet-pytorch/blob/54c869c51e0e3e12f7f92f551cdd2ecd164e2443/models/utils/detect_face.py#L358)
        margin = [
            _margin * (box[2] - box[0]) / (cropped_img_size - _margin),
            _margin * (box[3] - box[1]) / (cropped_img_size - _margin),
        ]
        raw_image_size = img.size
        box = [
            int(max(box[0] - margin[0] / 2, 0)),
            int(max(box[1] - margin[1] / 2, 0)),
            int(min(box[2] + margin[0] / 2, raw_image_size[0])),
            int(min(box[3] + margin[1] / 2, raw_image_size[1])),
        ]
        ret_boxes.append(box)

    return ret_faces, ret_boxes, boxes[-1]

def detect_and_save_faces(mp4_path, start_i, save_dir):
    reader = cv2.VideoCapture(mp4_path)
    # fps = reader.get(cv2.CAP_PROP_FPS)
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    images = []
    previous_box = None

    # print('Reading %s, extracting faces, and saving images' % mp4_path)
    for i in range(n_frames):
        _, image = reader.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(images) < filter_length:
            images.append(image)
        # else, detect faces in sequence and create new list
        else:
            face_images, boxes, previous_box = get_faces(detector, images, previous_box)
            save_images(tensor2npimage(face_images), save_dir, start_i)

            start_i += len(images)
            images = [image]
    # last sequence
    face_images, boxes, _ = get_faces(detector, images, previous_box)
    save_images(tensor2npimage(face_images), save_dir, start_i)

    start_i += len(images)

    reader.release()
    return start_i

def process_video(video_path):
    save_dir = video_path.rsplit('.')[0].replace('video', 'image')
    if not os.path.exists(save_dir):
        detect_and_save_faces(video_path, 0, save_dir)

def main():
    with open(args.paired_info) as f:
        paired_info = json.load(f)

    save_path = os.path.join(args.data_root, 'aligned_path.json')
    if os.path.exists(save_path):
        with open(save_path) as f:
            aligned_path = json.load(f)
    else:
        aligned_path = {}
    for actor in args.actors:
        num_videos = len(paired_info[actor]['neutral'])
        aligned_path[actor] = {}
        for emo in args.emotions:
            if emo == 'neutral':
                for i in tqdm(range(num_videos), desc=f'{actor} {emo}'):
                    video_path = os.path.join(args.data_root, actor, 'video/front/neutral/level_1', paired_info[actor]['neutral'][i]+'.mp4')
                    if not os.path.exists(video_path):
                        print(f'warning: file not exists {video_path}')
                    else:
                        process_video(video_path)
                continue
            aligned_path[actor][emo] = {}
            for i in tqdm(range(num_videos), desc=f'{actor} {emo}'):
                video_path = os.path.join(args.data_root, actor, 'video/front/', emo, f'level_{emo_2_lvl[emo]}', paired_info[actor][emo][i]+'.mp4')
                if not os.path.exists(video_path):
                    print(f'warning: file not exists {video_path}')
                else:
                    process_video(video_path)
                    name1 = paired_info[actor]['neutral'][i]
                    audio1 = os.path.join(args.data_root, actor, 'audio', 'neutral', 'level_1', name1+'.m4a')
                    name2 = paired_info[actor][emo][i]
                    audio2 = os.path.join(args.data_root, actor, 'audio', emo, f'level_{emo_2_lvl[emo]}', name2+'.m4a')
                    aligned_path[actor][emo][name1+'_'+name2] = list(map(lambda x: x.tolist(), align_audio(audio1, audio2)))
    
    with open(save_path, 'w') as f:
        json.dump(aligned_path, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',type=str,required=True, help='data root')
    parser.add_argument('--paired_info',type=str,required=True, help='paired_info.json')
    parser.add_argument('--actors', type=str, nargs='+', help='Subset of the MEAD actors', default=['M003'])
    parser.add_argument('--emotions', type=str, nargs='+', help='Selection of Emotions', default=['neutral', 'happy'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = MTCNN(image_size=cropped_img_size, margin=_margin, post_process=False, device=device)

    main()
