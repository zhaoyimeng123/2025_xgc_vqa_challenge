import os

import pandas as pd
import scipy.io as scio
import skvideo.io
from PIL import Image
import imageio

def extract_frame(videos_dir, video_name, save_folder):

    video_height = 1080  # the heigh of frames
    video_width = 1920  # the width of frames

    filename = os.path.join(videos_dir, video_name)

    reader = imageio.get_reader(filename, 'ffmpeg', fps=30, input_params=['-s', '1920x1080', '-pix_fmt', 'yuv420p'])
    for frame in reader:
        # 逐帧处理
        print(frame.shape)

    video_data = skvideo.io.vread(filename, video_height, video_width,
                                  inputdict={'-pix_fmt': 'yuvj420p'})

    video_name_str = video_name[:-4]

    video_length = video_data.shape[0]
    video_frame_rate = video_data.shape[0] // 15

    print(filename)
    print(video_length)
    print(video_frame_rate)


    video_read_index = 0

    frame_idx = 0

    video_length_min = 15

    for i in range(video_length):
        frame = video_data[i]
        if (video_read_index < video_length) and (frame_idx % (int(video_frame_rate)) == int(video_frame_rate/2)):
            read_frame = frame
            read_frame = Image.fromarray(read_frame)
            exit_folder(os.path.join(save_folder, video_name_str))
            read_frame.save(os.path.join(save_folder, video_name_str,
                                     '{:03d}'.format(video_read_index) + '.png'))
            video_read_index += 1
        frame_idx += 1


    if video_read_index < video_length_min:
        for i in range(video_read_index, video_length_min):
            read_frame.save(os.path.join(save_folder, video_name_str,
                                     '{:03d}'.format(i) + '.png'))

    return


def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return

if __name__ == '__main__':
    videos_dir = '/data/dataset/LIVE-QualcommDatabase/videos'
    filename_path = '/data/dataset/LIVE-QualcommDatabase/qualcommSubjectiveData.mat'
    save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/livequalcomm_image_all_fps1'

    m = scio.loadmat(filename_path)
    # pdb.set_trace()
    dataInfo = pd.DataFrame(m['qualcommVideoData'][0][0][0])
    dataInfo['MOS'] = m['qualcommSubjectiveData'][0][0][0]
    dataInfo.columns = ['file_names', 'MOS']
    dataInfo['file_names'] = dataInfo['file_names'].astype(str)
    dataInfo['file_names'] = dataInfo['file_names'].str.strip("[']")

    video_names = dataInfo['file_names'].tolist()

    n_video = len(video_names)

    for i in range(n_video):
        video_name = video_names[i]
        print('start extract {}th video: {}'.format(i, video_name))
        extract_frame(videos_dir, video_name, save_folder)
