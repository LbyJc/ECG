import scipy
import math
import random
import argparse
from time import time
import sys
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import misc
from misc import NativeScalerWithGradNormCount as NativeScaler
import pickle
import neurokit2 as nk
import matplotlib
import os
import torch
import torch.nn.functional as F
import scipy.io as sio
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import time
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
#from CustomLoss import CrossCorrelationAlignmentLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = StandardScaler()

# 体素网格的维度和范围
voxel_grid_dimensions = (9, 9, 17)
voxel_range = {
    "x": (-0.4, 0.4),
    "y": (0.35, 0.6),
    "z": (-0.4, 0.4),
}
# 体素步长计算
voxel_step = {
    "x": (voxel_range["x"][1] - voxel_range["x"][0]) / (voxel_grid_dimensions[0] - 1),
    "y": (voxel_range["y"][1] - voxel_range["y"][0]) / (voxel_grid_dimensions[1] - 1),
    "z": (voxel_range["z"][1] - voxel_range["z"][0]) / (voxel_grid_dimensions[2] - 1),
}

def cos_loss(pred, target):#余弦相似度损失（只关注波形不关注幅度）
    """
    pred:  (batch_size, seq_length)
    target:(batch_size, seq_length)
    """
    # shape=(batch_size,)
    cos_sim = F.cosine_similarity(pred, target, dim=1)
    
    loss = 1 - cos_sim.mean()
    return loss

# 定义数据加载和预处理函数
def load_files(directory):
    """
    从指定的文件夹中加载所有的 .pkl 文件。
    返回加载后的数据字典列表。
    """
    data_list = []
    
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(directory):
        if file_name.endswith('.mat') and (file_name.endswith('_1.mat') or file_name.endswith('_2.mat')):
            file_path = os.path.join(directory, file_name)
            # 使用 pickle 加载 .pkl 文件
            with open(file_path, 'rb') as f:
                data = sio.loadmat(f)
                data = data['data']
                data_list.append(data)
    
    return data_list

def load_files_from_subfolders(base_directory):
    """
    从指定的根文件夹（base_directory）及其所有子文件夹中加载所有的 .mat 文件。
    返回加载后的数据字典列表。
    """
    data_list = []
    
    # 遍历根目录下的所有子文件夹（如0326, 0331等）
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        
        # 确保是文件夹
        if os.path.isdir(folder_path):
            # 使用 os.walk 遍历每个子文件夹及其中的文件
            for root, dirs, files in os.walk(folder_path):
                for file_name in files:
                    if file_name.endswith(('_1.mat', '_2.mat', '_3.mat')):
                        file_path = os.path.join(root, file_name)
                        # 加载 .mat 文件
                        data = sio.loadmat(file_path)
                        data = data['data']  # 假设mat文件中的数据存储在 'data' 键下
                        data_list.append(data)
    
    return data_list

# def quantization(data, num_levels=256):
#     norm = (data - np.min(data)) / (np.max(data) - np.min(data))
#     quantized_data = np.round(norm * (num_levels - 1))
#     return quantized_data

def min_max_normalization(data):
    min_val = np.min(data, axis=(0, 1), keepdims=True)
    max_val = np.max(data, axis=(0, 1), keepdims=True)
    normalized_data = (data - min_val) / (max_val - min_val)
    normalized_data = normalized_data * 2 - 1
    return normalized_data

def min_max_normalization_ecg(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    normalized_data = normalized_data * 2 - 1
    return normalized_data


def sliding_window_sampling(time_series, sample_size, step):
    num_samples = (time_series.shape[1] - sample_size) // step + 1
    samples = [time_series[:, i * step : i * step + sample_size] for i in range(num_samples)]
    return np.array(samples)

def sliding_window_sampling1d(time_series, sample_size, step):
    num_samples = (len(time_series) - sample_size) // step + 1
    samples = [time_series[i * step : i * step + sample_size] for i in range(num_samples)]
    return np.array(samples)

def restore_real_coordinates(index_array):
    """
    将 pozXYZ 的体素网格索引转换为真实的物理坐标。
    """
    voxel_positions_real = np.zeros_like(index_array)

    voxel_positions_real[:, 0] = voxel_range["x"][0] + index_array[:, 0] * voxel_step["x"]
    voxel_positions_real[:, 1] = voxel_range["y"][0] + index_array[:, 1] * voxel_step["y"]
    voxel_positions_real[:, 2] = voxel_range["z"][0] + index_array[:, 2] * voxel_step["z"]
    
    return voxel_positions_real

def preprocess_data(data_dict, rcg_sample_size=640, ecg_sample_size=640, step=30, sampling_rate=200):
    """
    对单个数据字典进行预处理。
    """
    # 处理 RCG 数据
    #rcg_data = data_dict[0][0][0].astype(np.float32)
    rcg_data = data_dict[0][0].astype(np.float32)
    #rcg_data = min_max_normalization(rcg_data)
    rcg_data = scaler.fit_transform(rcg_data)
    RCG_segmented = rcg_data.transpose()
    RCG_segmented = np.expand_dims(RCG_segmented, axis=0)  # 增加一个维度
    # RCG_segmented = sliding_window_sampling(np.transpose(rcg_data), rcg_sample_size, step)
    # RCG_segmented = np.expand_dims(RCG_segmented, axis=2)  # 增加一个维度

    # 处理 ECG 数据
    #ecg_data = data_dict[0][0][1].astype(np.float32)
    ecg_data = data_dict[0][1].astype(np.float32).squeeze()


    # 量化清理后的 ECG 数据
    #ECG_quantized = quantization(ecg_clean)
    #ecg_norm = min_max_normalization_ecg(ecg_clean)
    #ecg_norm = scaler.fit_transform(ecg_data).transpose().squeeze()
    # 使用 neurokit2 进行 ECG 信号清理
    ecg_clean = nk.ecg_clean(ecg_data, sampling_rate=sampling_rate)
    ECG_segmented = np.expand_dims(ecg_clean, axis=0) # 增加一个维度
    # ECG_segmented = sliding_window_sampling1d(ecg_clean, ecg_sample_size, step)

    # 还原 pozXYZ 的真实坐标
    #pozXYZ_data = data_dict[0][0][2].astype(np.float32)
    pozXYZ_data = data_dict[0][2].astype(np.float32)
    pozXYZ_real = restore_real_coordinates(pozXYZ_data)  

    # 确保 pozXYZ 的重复次数与 ECG_segmented 的第一维度匹配
    num_segments = ECG_segmented.shape[0]
    pozXYZ_repeated = np.tile(pozXYZ_real, (num_segments, 1, 1))

    return RCG_segmented, ECG_segmented, pozXYZ_repeated






# 主函数：加载所有文件并预处理
def Preprocessing(directory, sampling_rate=200):
    """
    加载目录下的所有 .pkl 文件，并对其进行预处理。
    返回预处理后的 RCG、ECG 和 pozXYZ 数据列表。
    """
    # 加载所有 .pkl 文件
    #all_data = load_files(directory)
    all_data = load_files_from_subfolders(directory)
    
    # 初始化空列表，用于存储处理后的数据
    all_RCG = []
    all_ECG = []
    all_pozXYZ = []

    # 对每个数据字典进行预处理
    for data_dict in all_data:
        RCG_segmented, ECG_segmented, pozXYZ_segmented = preprocess_data(data_dict, sampling_rate=sampling_rate)
        
        # 展开并拼接结果
        all_RCG.append(RCG_segmented)
        all_ECG.append(ECG_segmented)
        all_pozXYZ.append(pozXYZ_segmented)

    # 将所有文件处理后的结果拼接成统一的数组
    all_RCG = np.concatenate(all_RCG, axis=0)
    all_ECG = np.concatenate(all_ECG, axis=0)
    all_pozXYZ = np.concatenate(all_pozXYZ, axis=0)

    return all_RCG, all_ECG, torch.tensor(all_pozXYZ, dtype=torch.float32)


def build_dataset(is_train, args):
    #path = os.path.join(args.root_path, 'Dataset_Aligned_BadFileDeleted' if is_train else 'Dataset_Aligned_BadFileDeleted_test')
    path = os.path.join(args.root_path, r'transfer_3376759_files_d8379d48\theTrainingData' if is_train else r'transfer_3376759_files_d8379d48\theTestingData')
    RCG, ECG, pozXYZ = Preprocessing(path)
    # 打印处理后的形状
    if is_train:
        print("Training RCG shape:", RCG.shape) 
        print("Training ECG shape:", ECG.shape)  
        print("Training pozXYZ shape:", pozXYZ.shape)
    else:
        print("Testing RCG shape:", RCG.shape) 
        print("Testing ECG shape:", ECG.shape)  
        print("Testing pozXYZ shape:", pozXYZ.shape)
    dataset = TensorDataset(torch.tensor(RCG, dtype=torch.float32), torch.tensor(ECG, dtype=torch.float32), pozXYZ)

    return dataset

def get_args_parser():
    parser = argparse.ArgumentParser('AirECG Training', add_help=False)
    
    # Training parameters
    parser.add_argument('--batch_size', default=1, type=int, 
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=101, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, 
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.0002, #origin 0.0001
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--lr', type=float, default=0.0001, #0.001
                        metavar='LR', 
                        help='learning rate (absolute lr)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='lr scheduler (default: cosine)')
    
    # Path parameters
    parser.add_argument('--root_path', default='D:\Code\Python\ECG', 
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_dir', default='./output_dir_pretrained', 
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_pretrained', 
                        help='path where to tensorboard log')

    # Resume training
    parser.add_argument('--resume', default='', 
                        help='resume from checkpoint')

    # Miscellaneous
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N', 
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true', 
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest="pin_mem")
    
    parser.set_defaults(pin_mem=True)
    
    return parser
a = nn.Parameter(torch.tensor(0.5))


parser = get_args_parser()
args, unkowns = parser.parse_known_args()

dataset_train = build_dataset(is_train=True, args=args)
dataset_val = build_dataset(is_train=False, args=args)

data_loader_train = torch.utils.data.DataLoader(
            dataset_train, shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

data_loader_val = torch.utils.data.DataLoader(
            dataset_val, shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
for i, [input, labels, pos] in enumerate(data_loader_val):
    print(f"Batch {i}:input shape: {input.shape}, label shape: {labels.shape}, pos shape: {pos.shape}")

import matplotlib.pyplot as plt
import torch

iterater = iter(data_loader_train)
for i in range(57):
    batch = next(iterater)
    inputs, labels, pos = batch


    label_batch = labels.detach().cpu().numpy()
    label_batch = scaler.fit_transform(label_batch.transpose()).transpose()
    input_batch = inputs.detach().cpu().numpy()


    plt.figure(figsize=(16, 4))
    label0 = label_batch[0, :]
    inputMean = input_batch.mean(axis=1).squeeze()[:]
    plt.plot(label0, label=f'ECG (label){i+1}')
    plt.plot(inputMean, label=f'RCG (input){i+1}')
    plt.xlabel('time step')
    plt.ylabel('amplitude')
    plt.legend()
    plt.title(f'ECG Input vs. Label (Sample {i+1})')
    plt.show()

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks


    def align_signals(ecg_signal, rcg_signal, prominence:float):
        
        ecg_peaks, _ = find_peaks(ecg_signal, prominence=prominence)
        rcg_peaks, _ = find_peaks(rcg_signal, prominence=prominence)

        if len(ecg_peaks) == 0 or len(rcg_peaks) == 0:
            raise ValueError("没有在ECG或RCG信号中找到波峰")

        
        ecg_first_peak = ecg_peaks[0]
        rcg_first_peak = rcg_peaks[0]

        
        offset = ecg_first_peak - rcg_first_peak
        print(f"Offset: {offset}")

        
        if offset > 0:
            # 将RCG信号向右平移
            rcg_shifted = rcg_signal[:-offset]  
            rcg_shifted = np.pad(rcg_shifted, (len(ecg_signal) - len(rcg_shifted), 0), 'constant')#补零
            #ecg_signal = ecg_signal[offset:]
            return ecg_signal, rcg_shifted
        else:
            # 将RCG信号向左平移
            rcg_shifted = rcg_signal[-offset:]  
            rcg_shifted = np.pad(rcg_shifted, (0, len(ecg_signal) - len(rcg_shifted)), 'constant')
            #ecg_signal = ecg_signal[:-offset]
            return ecg_signal, rcg_shifted

    ecg_signal= label_batch[0, 0:1000]
    rcg_signal= input_batch.mean(axis=1).squeeze()[0:1000]
    # ecg_signal= label_batch[0, 10000:11000]
    # rcg_signal= input_batch.mean(axis=1).squeeze()[0, 10000:11000]


    #aligned_ecg, aligned_rcg = align_signals(ecg_signal, rcg_signal, prominence=0.5)
    def findShift(ecg_signal, rcg_signal, prominence:float = 4.0):
        
        ecg_peaks, _ = find_peaks(ecg_signal, prominence=prominence)
        rcg_peaks, _ = find_peaks(rcg_signal, prominence=prominence)

        while len(ecg_peaks) == 0 or len(rcg_peaks) == 0:
            prominence -= 0.1
            if prominence <= 1:
                raise ValueError("没有在ECG或RCG信号中找到波峰")
            ecg_peaks, _ = find_peaks(ecg_signal, prominence=prominence)
            rcg_peaks, _ = find_peaks(rcg_signal, prominence=prominence)


        
        ecg_first_peak = ecg_peaks[0]
        rcg_first_peak = rcg_peaks[0]

        
        offset = ecg_first_peak - rcg_first_peak

        return offset
    


    def find_best_shift(ecg_signal, rcg_signal, max_shift=100, tolerance=5, init_prominence=4.0, min_prominence=1.0):
        """
        输入：ecg/rcg 信号，一维，最大平移范围，峰值容忍误差，初始prominence
        输出：最佳 shift，使得峰值重合最多
        """

        prominence = init_prominence

        
        while prominence >= min_prominence:
            ecg_peaks, _ = find_peaks(ecg_signal, prominence=prominence)
            rcg_peaks, _ = find_peaks(rcg_signal, prominence=prominence)

            if len(ecg_peaks) > 0 and len(rcg_peaks) > 0:
                break  

            prominence -= 0.1  

        if len(ecg_peaks) == 0 or len(rcg_peaks) == 0:
            raise ValueError("ECG 或 RCG 中找不到足够明显的波峰")

        
        best_shift = 0
        max_match = 0

        for shift in range(-max_shift, max_shift + 1):
            shifted_rcg_peaks = rcg_peaks + shift
           
            shifted_rcg_peaks = shifted_rcg_peaks[(shifted_rcg_peaks >= 0) & (shifted_rcg_peaks < len(rcg_signal))]

            # number of matches
            matches = 0
            for p in ecg_peaks:
                if np.any(np.abs(shifted_rcg_peaks - p) <= tolerance): #diff smaller than tolerance -> match + 1
                    matches += 1

            if matches > max_match:
                max_match = matches
                best_shift = shift

        return best_shift



    #shiftIndex = findShift(label_batch[0], input_batch.mean(axis=1)[0], prominence=4) # the diff = ecg_first_peak - rcg_first_peak
    shiftIndex = find_best_shift(label_batch[0], input_batch.mean(axis=1)[0], max_shift=100, tolerance=10, init_prominence=4.0, min_prominence=1.0)

    if shiftIndex > 0:
        aligned_ecg = label_batch[:, shiftIndex:]
        aligned_rcg = input_batch[:,:,:-shiftIndex,]
    elif shiftIndex < 0:
        aligned_ecg = label_batch[: , :shiftIndex]
        aligned_rcg = input_batch[:,:, -shiftIndex:,]
    else:
        pass

    #aligned_ecg, aligned_rcg = align_signals(label_batch[0], input_batch.mean(axis=1)[0], prominence=3.0)
    # print(f"input shape: {input_batch.shape}")
    # print(f"label shape: {label_batch.shape}")
    # print(f"aligned_ecg shape: {aligned_ecg.shape}")
    # print(f"aligned_rcg shape: {aligned_rcg.shape}")
    ecgAligned = aligned_ecg[0, :]
    rcgAligned = aligned_rcg.mean(axis=1)[0, :]
    # ecgAligned = aligned_ecg[10000:11000]
    # rcgAligned = aligned_rcg[10000:11000]

    # 绘制结果
    plt.figure(figsize=(16, 4))
    plt.plot(ecgAligned, label=f"ECG (label){i+1}", color='blue')
    plt.plot(rcgAligned, label=f"RCG (input){i+1}", color='orange')
    plt.xlabel("time step")
    plt.ylabel("amplitude")
    plt.legend()
    plt.title(f"ECG Input vs. Label (Aligned{i+1})")
    plt.show()