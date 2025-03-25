import numpy as np
import os
import pandas as pd
import cv2
import scipy.io as scio
import pdb
import skvideo.io


def extract_frame_clip(videos_dir, video_name, save_folder):
    filename = os.path.join(videos_dir, video_name)
    video_name_str = video_name[:-4]

    if os.path.exists(os.path.join(save_folder, video_name_str)):
        print('文件已存在：', video_name_str)
        return

    if video_name_str == '298':
        print('298视频有问题')
        return

    video_capture = cv2.VideoCapture()
    video_capture.open(filename)
    cap = cv2.VideoCapture(filename)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('The video length is ' + str(video_length))

    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # the heigh of frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # the width of frames
    print('The video height is ' + str(video_height))
    print('The video width is ' + str(video_width))

    video_read_index = 0

    frame_idx = 0

    video_length_min = 10
    # video_length_min = video_length

    n_interval = int(video_length / video_length_min)

    if n_interval > 0:

        for i in range(video_length):
            has_frames, frame = video_capture.read()  # 298视频有问题
            if has_frames:
                # key frame
                # if video_read_index < video_length:  # 抽取全部帧
                if (video_read_index < video_length) and (frame_idx % n_interval == int(n_interval / 2)):
                    read_frame = frame

                    # 居中裁剪操作
                    if video_height > video_width:
                        crop_size = video_width
                        start_y = (video_height - crop_size) // 2
                        start_x = 0
                    else:
                        crop_size = video_height
                        start_x = (video_width - crop_size) // 2
                        start_y = 0

                    read_frame = read_frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

                    exit_folder(os.path.join(save_folder, video_name_str))
                    cv2.imwrite(os.path.join(save_folder, video_name_str,
                                             '{:03d}'.format(video_read_index) + '.png'), read_frame)
                    video_read_index += 1
                frame_idx += 1

        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                cv2.imwrite(os.path.join(save_folder, video_name_str,
                                         '{:03d}'.format(i) + '.png'), read_frame)

    else:
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length):
                    read_frame = frame
                    # read_frame = frame
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


def extract_frame(videos_dir, video_name, save_folder, resize):
    filename = os.path.join(videos_dir, video_name)
    video_name_str = video_name[:-4]

    if os.path.exists(os.path.join(save_folder, video_name_str)):
        print('文件已存在：', video_name_str)
        return

    if video_name_str == '298':
        print('298视频有问题')
        return

    video_capture = cv2.VideoCapture()
    video_capture.open(filename)
    cap = cv2.VideoCapture(filename)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('The video length is ' + str(video_length))

    
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # the heigh of frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # the width of frames
    print('The video height is ' + str(video_height))
    print('The video width is ' + str(video_width))
    
    if video_height > video_width:
        video_width_resize = resize
        video_height_resize = int(video_width_resize/video_width*video_height)
    else:
        video_height_resize = resize
        video_width_resize = int(video_height_resize/video_height*video_width)
        
    dim = (video_width_resize, video_height_resize)

    video_read_index = 0

    frame_idx = 0
    
    video_length_min = 10
    # video_length_min = video_length

    n_interval = int(video_length/video_length_min)

    if n_interval > 0:

        for i in range(video_length):
            has_frames, frame = video_capture.read()  # 298视频有问题
            if has_frames:
                # key frame
                # if video_read_index < video_length:  # 抽取全部帧
                if (video_read_index < video_length) and (frame_idx % n_interval == int(n_interval/2)):
                    read_frame = cv2.resize(frame, dim)
                    # read_frame = frame
                    exit_folder(os.path.join(save_folder, video_name_str))
                    cv2.imwrite(os.path.join(save_folder, video_name_str,
                                             '{:03d}'.format(video_read_index) + '.png'), read_frame)
                    video_read_index += 1
                frame_idx += 1
                
        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                cv2.imwrite(os.path.join(save_folder, video_name_str,
                                         '{:03d}'.format(i) + '.png'), read_frame)

    else:
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length):
                    read_frame = cv2.resize(frame, dim)
                    # read_frame = frame
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

    videos_dir = '/data/dataset/LBVD/videos'
    filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_data.mat'
    save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_image_all_fps1_clip'

    dataInfo = scio.loadmat(filename_path)
    n_video = len(dataInfo['video_names'])
    video_names = []

    for i in range(n_video):
        video_names.append(dataInfo['video_names'][i][0][0])

    n_video = len(video_names)

    for i in range(n_video):
        video_name = video_names[i]
        print('start extract {}th video: {}'.format(i, video_name))
        extract_frame_clip(videos_dir, video_name, save_folder)
