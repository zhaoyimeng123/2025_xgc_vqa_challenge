import numpy as np
import os
import pandas as pd
import cv2
import scipy.io as scio


def extract_frame(videos_dir, video_name, save_folder):
    filename = os.path.join(videos_dir, video_name)
    video_name_str = video_name[:-4]
    video_capture = cv2.VideoCapture()
    video_capture.open(filename)
    cap = cv2.VideoCapture(filename)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # the heigh of frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # the width of frames

    video_read_index = 0

    frame_idx = 0

    video_length_min = 8

    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if has_frames:
            # key frame
            if (video_read_index < video_length) and (frame_idx % (int(video_frame_rate)) == int(video_frame_rate / 2)):
                read_frame = frame
                exit_folder(os.path.join(save_folder, video_name_str))
                cv2.imwrite(os.path.join(save_folder, video_name_str,
                                         '{:03d}'.format(video_read_index) + '.png'), read_frame)
                video_read_index += 1
            frame_idx += 1

    if video_read_index < video_length_min:
        for i in range(video_read_index, video_length_min):
            cv2.imwrite(os.path.join(save_folder, video_name_str,
                                     '{:03d}'.format(i) + '.png'), read_frame)

    return


def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return


if __name__ == '__main__':
    videos_dir = '/data/dataset/KoNViD_1k_videos'
    filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/KoNViD-1k_data.mat'

    dataInfo = scio.loadmat(filename_path)
    n_video = len(dataInfo['video_names'])
    video_names = []

    for i in range(n_video):
        video_names.append(dataInfo['video_names'][i][0][0])

    save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_image_all_fps1'
    for i in range(n_video):
        video_name = video_names[i]
        v_name = video_name.split('_')[0] + '.mp4'
        print('start extract {}th video: {}'.format(i, video_name))
        extract_frame(videos_dir, v_name, save_folder)
