# 2025_xgc_vqa_challenge

### 一、data prepare

#### 1、extract frames

```shell
python extract_frame/extract_frame_xgc.py
```

#### 2、extract distortion frature

```shell
use [Re-IQA](https://github.com/avinabsaha/ReIQA/blob/main/demo_quality_aware_feats.py) extract distortion features
```

```python
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import csv
import os
import numpy as np
import pandas as pd
import scipy.io as scio
import torch
import torch.utils.data.distributed
from PIL import Image
from torchvision import transforms

from networks.build_backbone import build_model
from options.train_options import TrainOptions


def read_float_with_comma(num):
    return float(num.replace(",", "."))
def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return

def run_inference(img_path):
    args = TrainOptions().parse()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # build model
    model, _ = build_model(args)
    model = torch.nn.DataParallel(model)

    # check and resume a model
    ckpt_path = './re-iqa_ckpts/quality_aware_r50.pth'

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model.to(args.device)
    model.eval()



    image = Image.open(img_path).convert('RGB')

    image2 = image.resize((image.size[0] // 2, image.size[1] // 2))  # half-scale

    # transform to tensor
    img1 = transforms.ToTensor()(image).unsqueeze(0)
    img2 = transforms.ToTensor()(image2).unsqueeze(0)

    # # 计算 FLOPs
    # from fvcore.nn import FlopCountAnalysis
    # # input_tensor = torch.randn(1, 3, 224, 224)
    # flop_analyzer = FlopCountAnalysis(model, img1.to(args.device))
    # flops = flop_analyzer.total()
    # # 检查是否低于 120G FLOPs
    # print(f"Total FLOPs: {flops:.2e}")  # 以科学计数法输出
    # print('Total GFLOPs: %.2f ' % (flops / 1e9))

    with torch.no_grad():
        feat1 = model.module.encoder(img1.to(args.device))  # (1, 2048)
        feat2 = model.module.encoder(img2.to(args.device))  # (1, 2048)
        feat = torch.cat((feat1, feat2), dim=1).detach().cpu().numpy()  # (1, 4096)

    # save features 
    # save_path = "feats_quality_aware/"
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # np.save("feats_quality_aware/" + img_path[img_path.rfind("/") + 1:-4] + '_quality_aware_features.npy', feat)
    # print('Quality Aware feature Extracted')
    return feat


if __name__ == '__main__':
    # img_path = "./sample_images/10004473376.jpg"
    # run_inference(img_path)
    database = 'xgc'
    video_names = []
    if database == 'KoNViD-1k':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_image_all_fps1'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_dist_quality_aware'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/KoNViD-1k_data.mat'
        dataInfo = scio.loadmat(filename_path)
        n_video = len(dataInfo['video_names'])
        for i in range(n_video):
            video_names.append(dataInfo['video_names'][i][0][0].split('_')[0])
        video_length_read = 8
    elif database == 'LBVD':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_image_all_fps1_clip'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/lbvd_dist_quality_aware_clip'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_data.mat'
        dataInfo = scio.loadmat(filename_path)
        n_video = len(dataInfo['video_names'])
        for i in range(n_video):
            video_names.append(dataInfo['video_names'][i][0][0])

        if '298.mp4' in video_names:
            video_names.remove('298.mp4')
            n_video = n_video - 1
        # video_length_read = 8
        video_length_read = 4
        frame_start = 3  # 标记从哪一帧开始提取

    elif database == 'LSVQ_train':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_image_all_fps1_motion'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_dist_quality_aware'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_train.csv'

        column_names = ['name', 'p1', 'p2', 'p3',
                        'height', 'width', 'mos_p1',
                        'mos_p2', 'mos_p3', 'mos',
                        'frame_number', 'fn_last_frame', 'left_p1',
                        'right_p1', 'top_p1', 'bottom_p1',
                        'start_p1', 'end_p1', 'left_p2',
                        'right_p2', 'top_p2', 'bottom_p2',
                        'start_p2', 'end_p2', 'left_p3',
                        'right_p3', 'top_p3', 'bottom_p3',
                        'start_p3', 'end_p3', 'top_vid',
                        'left_vid', 'bottom_vid', 'right_vid',
                        'start_vid', 'end_vid', 'is_test', 'is_valid']

        dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")
        video_names = dataInfo['name'].tolist()
        n_video = len(video_names)
        video_length_read = 8
    elif database == 'LSVQ_test':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_image_all_fps1_motion'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_dist_quality_aware/'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test.csv'

        column_names = ['name', 'p1', 'p2', 'p3',
                        'height', 'width', 'mos_p1',
                        'mos_p2', 'mos_p3', 'mos',
                        'frame_number', 'fn_last_frame', 'left_p1',
                        'right_p1', 'top_p1', 'bottom_p1',
                        'start_p1', 'end_p1', 'left_p2',
                        'right_p2', 'top_p2', 'bottom_p2',
                        'start_p2', 'end_p2', 'left_p3',
                        'right_p3', 'top_p3', 'bottom_p3',
                        'start_p3', 'end_p3', 'top_vid',
                        'left_vid', 'bottom_vid', 'right_vid',
                        'start_vid', 'end_vid', 'is_test', 'is_valid']

        dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")
        video_names = dataInfo['name'].tolist()
        n_video = len(video_names)
        video_length_read = 8
    elif database == 'LSVQ_test_1080p':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_image_all_fps1_motion'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_dist_quality_aware/'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test_1080p.csv'

        column_names = ['name', 'p1', 'p2', 'p3',
                        'height', 'width', 'mos_p1',
                        'mos_p2', 'mos_p3', 'mos',
                        'frame_number', 'fn_last_frame', 'left_p1',
                        'right_p1', 'top_p1', 'bottom_p1',
                        'start_p1', 'end_p1', 'left_p2',
                        'right_p2', 'top_p2', 'bottom_p2',
                        'start_p2', 'end_p2', 'left_p3',
                        'right_p3', 'top_p3', 'bottom_p3',
                        'start_p3', 'end_p3', 'top_vid',
                        'left_vid', 'bottom_vid', 'right_vid',
                        'start_vid', 'end_vid', 'is_valid']

        dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")
        video_names = dataInfo['name'].tolist()
        n_video = len(video_names)
        video_length_read = 8
    elif database == 'livevqc':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/livevqc_image_all_fps1'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/livevqc_dist_quality_aware'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/LiveVQC_data.mat'
        m = scio.loadmat(filename_path)
        # pdb.set_trace()tiva
        dataInfo = pd.DataFrame(m['video_list'])
        dataInfo.columns = ['file_names']
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
        video_names = dataInfo['file_names'].tolist()
        video_length_read = 8
    elif database == 'CVD2014':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/cvd2014_image_all_fps1'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/cvd2014_dist_quality_aware'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/CVD2014_Realignment_MOS.csv'
        file_names = []
        openfile = open(filename_path, 'r', newline='')
        lines = csv.DictReader(openfile, delimiter=';')

        for line in lines:
            if len(line['File_name']) > 0:
                file_names.append(line['File_name'])

        dataInfo = pd.DataFrame(file_names)
        dataInfo.columns = ['file_names']
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        video_names = dataInfo['file_names'].tolist()
        video_length_read = 10
    elif database == 'youtube_ugc':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/youtube_ugc_image_all_fps05'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/youtube_ugc_dist_quality_aware'
        filename_path = '/data/dataset/YoutubeYGC/original_videos_MOS_for_YouTube_UGC_dataset.xlsx'
        # 读取指定的工作表
        df = pd.read_excel(filename_path, sheet_name='diff2')
        # 读取指定列并将其转换为列表
        video_names = [f"{filename}_crf_10_ss_00_t_20.0.mp4" for filename in df['vid'].astype(str).tolist()]
        video_length_read = 10
    elif database == 'xgc':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_image_all_fps1'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_dist_quality_aware'
        filename_path = '/data/dataset/XGC-dataset/test.txt'
        # 读取 TXT 文件，假设是以 Tab（\t）分隔的
        dataInfo = pd.read_csv(filename_path, sep="\t", header=None)
        # 重命名列，第一列是视频文件名，后面列是 MOS 相关数据
        # dataInfo.columns = ['file_names', 'MOS1', 'MOS2', 'MOS3', 'MOS4', 'MOS5', 'MOS6']
        dataInfo.columns = ['file_names']
        # 确保文件名是字符串
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        # 获取视频文件名列表
        video_names = dataInfo['file_names'].tolist()
        # 计算视频数量
        n_video = len(video_names)
        video_length_read = 8
        frame_start = 0
    elif database == 'kvq_val':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/val/kvq_image_all_fps1_clip'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/val/kvq_dist_quality_aware_clip'
        filename_path = '/data/dataset/KVQ/groundtruth_label 2/truth.csv'
        # 读取 CSV 文件，假设是逗号（,）分隔的
        dataInfo = pd.read_csv(filename_path, sep=",", header=0)  # header=0 表示第一行为列名
        # 确保列名正确
        dataInfo.columns = ['file_names', 'score']
        # 确保文件名是字符串
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        # 获取视频文件名列表
        video_names = dataInfo['file_names'].tolist()
        video_length_read = 4
        frame_start = 2
    elif database == 'kvq_train':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/train/kvq_image_all_fps1_clip'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/train/kvq_dist_quality_aware_clip'
        filename_path = '/data/dataset/KVQ/train_data.csv'
        # 读取 CSV 文件，假设是逗号（,）分隔的
        dataInfo = pd.read_csv(filename_path, sep=",", header=0)  # header=0 表示第一行为列名
        # 确保列名正确
        dataInfo.columns = ['file_names', 'score']
        # 确保文件名是字符串
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        # 获取视频文件名列表
        video_names = dataInfo['file_names'].tolist()
        video_length_read = 6
        frame_start = 2

    for i in range(len(video_names)):
        video_name_str = video_names[i]
        if database == 'livevqc' or database == 'LBVD' or database == 'youtube_ugc' or database == 'xgc':
            video_name_str = video_name_str[0:-4]
        elif database == 'kvq_val' or database == 'kvq_train':
            video_name_str = video_name_str.split('/')[1]
            video_name_str = video_name_str[0:-4]
        path_name = os.path.join(imgs_dir, video_name_str)
        for j in range(video_length_read):
            if os.path.exists(os.path.join(save_folder, video_name_str, '{:03d}'.format(j+frame_start) + '.npy')):
                print(f"已存在：{os.path.join(save_folder, video_name_str, '{:03d}'.format(j+frame_start) + '.npy')}")
                continue
            imge_name = os.path.join(path_name, '{:03d}'.format(int(j+frame_start)) + '.png')
            print(i, imge_name)
            feat = run_inference(imge_name)
            exit_folder(os.path.join(save_folder, video_name_str))
            np.save(os.path.join(save_folder, video_name_str, '{:03d}'.format(j+frame_start)), feat)

```



#### 3、Extract motion features

```shell
use [VideoMae V2](https://github.com/OpenGVLab/VideoMAEv2/blob/master/extract_tad_feature.py) extract motion features
```

```python
"""Extract features for temporal action detection datasets"""
import argparse
import csv
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import cv2

import numpy as np
import pandas as pd
import torch
from timm.models import create_model
from torchvision import transforms

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
from dataset.loader import get_video_loader
import scipy.io as scio


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)


class ToFloatTensorInZeroOne(object):

    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


def get_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)

    parser.add_argument(
        '--data_set',
        default='TEST_VIDEO',
        choices=['THUMOS14', 'FINEACTION', 'TEST_VIDEO'],
        type=str,
        help='dataset')

    parser.add_argument(
        '--data_path',
        default=None,
        type=str,
        help='dataset path')
    parser.add_argument(
        '--save_path',
        default=None,
        type=str,
        help='path for saving features')

    parser.add_argument(
        '--model',
        default='vit_giant_patch14_224',
        type=str,
        metavar='MODEL',
        help='Name of model')
    parser.add_argument(
        '--ckpt_path',
        default='/data/user/zhaoyimeng/VideoMAEv2-master/vit_g_hybrid_pt_1200e_k710_ft.pth',
        help='load from checkpoint')

    return parser.parse_args()


def get_start_idx_range(data_set):
    def thumos14_range(num_frames):
        return range(0, num_frames - 15, 4)

    def fineaction_range(num_frames):
        return range(0, num_frames - 15, 16)

    def test_video_range(num_frames, video_frame_rate):
        return range(0, num_frames - video_frame_rate, video_frame_rate)

    if data_set == 'THUMOS14':
        return thumos14_range
    elif data_set == 'FINEACTION':
        return fineaction_range
    elif data_set == 'TEST_VIDEO':
        return test_video_range  # todo 每16帧作为一个clip，把两个相邻的clip合起来作为一个feature(对应slowfast中的32帧)，总共8个feature
    else:
        raise NotImplementedError()

def adjust_array_clip(arr):
    # 获取当前数组的第一维（行数）和第二维（列数）
    current_rows, cols = arr.shape
    if current_rows < 4:
        last_row = arr[-1]  # 获取最后一行
        rows_to_add = 4 - current_rows  # 需要补充的行数
        arr = np.vstack([arr, np.tile(last_row, (rows_to_add, 1))])  # 复制最后一行并堆叠
        return arr
    if current_rows < 8:
        # 如果行数小于 8，用最后一行进行复制，直到补足 8 行
        last_row = arr[-1]  # 获取最后一行
        rows_to_add = 8 - current_rows  # 需要补充的行数
        arr = np.vstack([arr, np.tile(last_row, (rows_to_add, 1))])  # 复制最后一行并堆叠
        arr = arr[2: 6, :]
    elif current_rows >= 8:
        # 如果行数大于 8，则截取前 8 行
        arr = arr[:8, :]
        # arr = arr[2: 6, :]
        arr = arr[3: 7, :]
    return arr

def adjust_array(arr):
    # 获取当前数组的第一维（行数）和第二维（列数）
    current_rows, cols = arr.shape
    if current_rows < 8:
        # 如果行数小于 8，用最后一行进行复制，直到补足 8 行
        last_row = arr[-1]  # 获取最后一行
        rows_to_add = 8 - current_rows  # 需要补充的行数
        arr = np.vstack([arr, np.tile(last_row, (rows_to_add, 1))])  # 复制最后一行并堆叠
    elif current_rows > 16:  # youtubeUGC
        arr = arr[::2][: 8]
    elif current_rows > 8:
        # 如果行数大于 8，则截取前 8 行
        arr = arr[:8, :]
    return arr


def extract_feature(args):
    dataset = 'xgc'
    if dataset == 'LSVQ_train':
        datainfo_train = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_train.csv'
        column_names = ['name', 'p1', 'p2', 'p3',
                        'height', 'width', 'mos_p1',
                        'mos_p2', 'mos_p3', 'mos',
                        'frame_number', 'fn_last_frame', 'left_p1',
                        'right_p1', 'top_p1', 'bottom_p1',
                        'start_p1', 'end_p1', 'left_p2',
                        'right_p2', 'top_p2', 'bottom_p2',
                        'start_p2', 'end_p2', 'left_p3',
                        'right_p3', 'top_p3', 'bottom_p3',
                        'start_p3', 'end_p3', 'top_vid',
                        'left_vid', 'bottom_vid', 'right_vid',
                        'start_vid', 'end_vid', 'is_test', 'is_valid']
        dataInfo = pd.read_csv(datainfo_train, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")
        video_names = dataInfo['name']
        # get video path
        # vid_list = [args.data_path + name for name in video_names]
        vid_list = video_names
    elif dataset == 'LSVQ_test':
        datainfo_train = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test.csv'
        column_names = ['name', 'p1', 'p2', 'p3',
                        'height', 'width', 'mos_p1',
                        'mos_p2', 'mos_p3', 'mos',
                        'frame_number', 'fn_last_frame', 'left_p1',
                        'right_p1', 'top_p1', 'bottom_p1',
                        'start_p1', 'end_p1', 'left_p2',
                        'right_p2', 'top_p2', 'bottom_p2',
                        'start_p2', 'end_p2', 'left_p3',
                        'right_p3', 'top_p3', 'bottom_p3',
                        'start_p3', 'end_p3', 'top_vid',
                        'left_vid', 'bottom_vid', 'right_vid',
                        'start_vid', 'end_vid', 'is_test', 'is_valid']
        dataInfo = pd.read_csv(datainfo_train, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")
        video_names = dataInfo['name']
        # get video path
        # vid_list = [args.data_path + name for name in video_names]
        vid_list = video_names
    elif dataset == 'LSVQ_test_1080p':
        datainfo_train = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test_1080p.csv'
        column_names = ['name', 'p1', 'p2', 'p3',
                        'height', 'width', 'mos_p1',
                        'mos_p2', 'mos_p3', 'mos',
                        'frame_number', 'fn_last_frame', 'left_p1',
                        'right_p1', 'top_p1', 'bottom_p1',
                        'start_p1', 'end_p1', 'left_p2',
                        'right_p2', 'top_p2', 'bottom_p2',
                        'start_p2', 'end_p2', 'left_p3',
                        'right_p3', 'top_p3', 'bottom_p3',
                        'start_p3', 'end_p3', 'top_vid',
                        'left_vid', 'bottom_vid', 'right_vid',
                        'start_vid', 'end_vid', 'is_valid']
        dataInfo = pd.read_csv(datainfo_train, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")
        video_names = dataInfo['name']
        # get video path
        # vid_list = [args.data_path + name for name in video_names]
        vid_list = video_names
    elif dataset == 'YoutubeUGC':
        # 定义 Excel 文件路径和表名、列名
        file_path = "/data/dataset/YoutubeYGC/original_videos_MOS_for_YouTube_UGC_dataset.xlsx"
        sheet_name = "diff2"  # 指定你要读取的表名
        column_name = "vid"  # 指定你要读取的列名
        df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[column_name])
        df_list = df[column_name].tolist()
        vid_list = [name + '_crf_10_ss_00_t_20.0.mp4' for name in df_list]
        print(vid_list)
    elif dataset == 'CVD2014':
        def read_float_with_comma(num):
            return float(num.replace(",", "."))

        file_names = []
        mos = []
        openfile = open("/data/user/zhaoyimeng/ModularBVQA/data/CVD2014_Realignment_MOS.csv", 'r', newline='')
        lines = csv.DictReader(openfile, delimiter=';')

        for line in lines:
            if len(line['File_name']) > 0:
                file_names.append(line['File_name'])
            if len(line['Realignment MOS']) > 0:
                mos.append(read_float_with_comma(line['Realignment MOS']))

        dataInfo = pd.DataFrame(file_names)
        dataInfo['MOS'] = mos
        dataInfo.columns = ['file_names', 'MOS']
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        dataInfo['file_names'] = dataInfo['file_names'] + ".avi"
        video_names = dataInfo['file_names'].tolist()
        vid_list = video_names
    elif dataset == 'KoNViD-1k':
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/KoNViD-1k_data.mat'

        dataInfo = scio.loadmat(filename_path)
        n_video = len(dataInfo['video_names'])
        video_names = []

        for i in range(n_video):
            video_names.append(dataInfo['video_names'][i][0][0])
        vid_list = [name.split('_')[0] + '.mp4' for name in video_names]
    elif dataset == 'LiveVQC':
        args.data_path = '/data/dataset/LIVE_VQC/Video'
        args.save_path = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/LiveVQC_VideoMAE_feat'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/LiveVQC_data.mat'
        m = scio.loadmat(filename_path)
        # pdb.set_trace()tiva
        dataInfo = pd.DataFrame(m['video_list'])
        dataInfo['MOS'] = m['mos']
        dataInfo.columns = ['file_names', 'MOS']
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
        video_names = dataInfo['file_names'].tolist()
        vid_list = video_names
    elif dataset == 'LBVD':
        args.data_path = '/data/dataset/LBVD/videos'
        args.save_path = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/LBVD_VideoMAE_feat_clip'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_data.mat'
        dataInfo = scio.loadmat(filename_path)
        n_video = len(dataInfo['video_names'])
        video_names = []
        for i in range(n_video):
            video_names.append(dataInfo['video_names'][i][0][0])
        vid_list = video_names
    elif dataset == 'xgc':
        args.data_path = '/data/dataset/XGC-dataset/test'
        args.save_path = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/xgc_VideoMAE_feat'
        filename_path = '/data/dataset/XGC-dataset/test.txt'
        # 读取 TXT 文件，假设是以 Tab（\t）分隔的
        dataInfo = pd.read_csv(filename_path, sep="\t", header=None)
        # 重命名列，第一列是视频文件名，后面列是 MOS 相关数据
        # dataInfo.columns = ['file_names', 'MOS1', 'MOS2', 'MOS3', 'MOS4', 'MOS5', 'MOS6']
        dataInfo.columns = ['file_names']
        # 确保文件名是字符串
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        # 获取视频文件名列表
        video_names = dataInfo['file_names'].tolist()
        # 计算视频数量
        n_video = len(video_names)
        vid_list = video_names
    elif dataset == 'kvq_val':
        args.data_path = '/data/dataset/KVQ/val'
        args.save_path = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/KVQ/val/kvq_VideoMAE_feat_clip'
        filename_path = '/data/dataset/KVQ/groundtruth_label 2/truth.csv'
        # 读取 CSV 文件，假设是逗号（,）分隔的
        dataInfo = pd.read_csv(filename_path, sep=",", header=0)  # header=0 表示第一行为列名
        # 确保列名正确
        dataInfo.columns = ['file_names', 'score']
        # 确保文件名是字符串
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        # 获取视频文件名列表
        video_names = dataInfo['file_names'].tolist()
        vid_list = video_names
    elif dataset == 'kvq_train':
        args.data_path = '/data/dataset/KVQ/train'
        args.save_path = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/KVQ/train/kvq_VideoMAE_feat_clip'
        filename_path = '/data/dataset/KVQ/train_data.csv'
        # 读取 CSV 文件，假设是逗号（,）分隔的
        dataInfo = pd.read_csv(filename_path, sep=",", header=0)  # header=0 表示第一行为列名
        # 确保列名正确
        dataInfo.columns = ['file_names', 'score']
        # 确保文件名是字符串
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        # 获取视频文件名列表
        video_names = dataInfo['file_names'].tolist()

        vid_list = video_names


        # preparation
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    video_loader = get_video_loader()


    transform = transforms.Compose(
        [ToFloatTensorInZeroOne(),
         Resize((224, 224))])

    # random.shuffle(vid_list)

    # get model & load ckpt
    model = create_model(
        args.model,
        img_size=224,
        pretrained=False,
        num_classes=710,
        all_frames=16,
        tubelet_size=2,
        drop_path_rate=0.3,
        use_mean_pooling=True)
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break
    model.load_state_dict(ckpt)
    model.eval()
    model.cuda()

    with torch.no_grad():
        # 计算 FLOPs
        from fvcore.nn import FlopCountAnalysis
        input_tensor = torch.randn(1, 3, 16, 224, 224)
        flop_analyzer = FlopCountAnalysis(model, input_tensor.cuda())
        flops = flop_analyzer.total()
        # 检查是否低于 120G FLOPs
        # print(f"Total FLOPs: {flops:.2e}")  # 以科学计数法输出
        print('Total GFLOPs: %.2f ' % (flops / 1e9))

    # extract feature
    num_videos = len(vid_list)
    for idx, vid_name in enumerate(vid_list):
        # url = os.path.join(args.save_path, vid_name.split('.')[0] + '.npy')
        if dataset == 'LSVQ_train' or dataset == 'LSVQ_test' or dataset == 'LSVQ_test_1080p':
            url = os.path.join(args.save_path, vid_name + '.npy')
            if not os.path.exists(args.save_path + '/' + vid_name.split('/')[0]):
                os.makedirs(args.save_path + '/' + vid_name.split('/')[0])
            vid_name = vid_name + '.mp4'
        elif dataset == 'YoutubeUGC' or dataset == 'CVD2014' or dataset == 'KoNViD-1k' or dataset == 'LiveVQC' or dataset == 'LBVD'\
                or dataset == 'xgc':
            url = os.path.join(args.save_path, vid_name[0:-4] + '.npy')
        elif dataset == 'kvq_val' or dataset == 'kvq_train':
            vid_name = vid_name.split('/')[1]
            url = os.path.join(args.save_path, vid_name[0:-4] + '.npy')
        if os.path.exists(url):
            print("exists " + url)
            continue

        video_path = os.path.join(args.data_path, vid_name)
        vr = video_loader(video_path)

        cap = cv2.VideoCapture(video_path)
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        if video_frame_rate == 0:
            print(vid_name, 'video_frame_rate = 0')
            continue

        start_idx_range = get_start_idx_range(args.data_set)

        feature_list = []
        for start_idx in start_idx_range(len(vr), video_frame_rate):
            if start_idx + 16 <= start_idx + video_frame_rate:  # 帧率>=16
                data = vr.get_batch(np.arange(start_idx, start_idx + 16)).asnumpy()
            else:
                # 获取可用的所有帧
                available_data = vr.get_batch(np.arange(start_idx, start_idx + video_frame_rate)).asnumpy()

                padding_frames = np.tile(available_data[-1:], (16 - video_frame_rate, 1, 1, 1))
                data = np.concatenate([available_data, padding_frames], axis=0)

            frame = torch.from_numpy(data)  # torch.Size([16, 566, 320, 3])
            frame_q = transform(frame)  # torch.Size([3, 16, 224, 224])
            input_data = frame_q.unsqueeze(0).cuda()

            with torch.no_grad():
                feature = model.forward_features(input_data)
                feature_list.append(feature.cpu().numpy())

           
            if len(feature_list) > 8:
                print("feature_list > 8", vid_name)
                break

        if len(feature_list) == 0:
            print(vid_name, 'feature_list = 0')
            continue

        # [N, C]
        feature_list_numpy = np.vstack(feature_list)

        feature_list_numpy_adjust = adjust_array(feature_list_numpy)

        print(f'feature_list_numpy_adjust shape: {feature_list_numpy_adjust.shape}')
        np.save(url, feature_list_numpy_adjust)

        print(f'[{idx + 1} / {num_videos}]: save feature on {url}')


if __name__ == '__main__':
    args = get_args()
    extract_feature(args)

```



### 二、train

#### 1、pretrain on LSVQ

```shell
python train_other_modular_dist_loda_cross_atten_videomae_v10_param_xgc.py  --database LSVQ
```

We also provide trained weights [pre-train on LSVQ](https://pan.baidu.com/s/1sGj7QL6vKRZH0xfAWrCpIg?pwd=xgcc 提取码: xgcc 
--来自百度网盘超级会员v5的分享) 

#### 2、fine-tune on xgc dataset

```shell
python train_other_modular_dist_loda_cross_atten_videomae_v10_param_xgc.py  --database xgc --trained_model pretrainLSVQ.pth
```

```markdown
Note: The MOS labels used in lines 145, 186, and 209 should be consistent. For example, line 145: labels = mos2.to(device).float()， 186 lines: label[i] = mos2.item()， Line 209: ..._dim_2_epoch_.... The weight of the second dimension is obtained by dim_2 epoch
```

We also provide trained weights on upon baidudisk link.

### 三、test

```shell
python test_baseline_modular_videomae_v10_xgc_split.py --trained_model xgc_round_0_dim1_SRCC_0.799520.pth
```

```markdown
Similarly, the weight of which dimension to load should be modified on line 61 to save the data
```

finally，` merge_txt.py `用于合并6个维度以得到最终结果。

-----------------------------

We originally planned to upload the extracted frames, distortion features, and motion features together to BaiduDisk, but the data is on the server and transmission is a bit troublesome, so we only uploaded the weights, and we provided the code for extracting features.

