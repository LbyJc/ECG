# 将ecg与rcg对齐功能融入。

import numpy as np
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import scipy.interpolate


from scipy.signal import resample, find_peaks
import neurokit2 as nk
import scipy.io as sio



def compute_second_derivative(data, h):
    second_derivative = np.zeros_like(data)
    N = len(data)

    for i in range(3, N-3):
        s0 = data[i]
        s_minus_3 = data[i-3]
        s_minus_2 = data[i-2]
        s_minus_1 = data[i-1]
        s_plus_1 = data[i+1]
        s_plus_2 = data[i+2]
        s_plus_3 = data[i+3]

        # 应用给定的二次微分公式
        second_derivative[i] = ((s_minus_3 + s_plus_3) +
            2 * (s_minus_2 + s_plus_2) -
            (s_minus_1 + s_plus_1) -
            4 * s0) / (16 * h**2)

    return second_derivative

def calculation(
        dataCube,
        voxel_position,
        idx_info,
        antenna_positions,
        info,
        c=299792458  # 光速
):
    # 初始化时域信号数组
    s = np.zeros(info["clip"], dtype=np.complex64)

    # 将antenna_positions转换为numpy数组
    antenna_positions_array = np.array([antenna_positions[channel] for channel in range(0, 12)])

    # 计算所有天线到体素位置的距离
    r = np.linalg.norm(antenna_positions_array - voxel_position, axis=1) * 2  # shape: (12,)

    # 创建时间轴
    t = np.linspace(2e-7, 51.2e-6, num=256)

    # 计算相位偏移，考虑天线1, 3, 5, 7, 9, 11的相位取反
    phase_shift = np.exp(1j * 2 * np.pi * info["freqSlope"] * r[:, np.newaxis] * t / c) * \
                  np.exp(1j * 2 * np.pi * r[:, np.newaxis] / info["waveLength"])

    # 天线1, 3, 5, 7, 9, 11的相位取反
    phase_shift[[0, 2, 4, 6, 8, 10], :] *= -1  # 0-based index

    # 将天线信号与相位偏移相乘并累加
    for frame_idx in range(info["clip"]):
        y_nt = dataCube[frame_idx, 0, :, :, :]  # shape: (3, 4, 256)
        y_nt = y_nt.reshape(-1, y_nt.shape[-1])  # shape: (12, 256)

        # 累加所有天线的信号
        s[frame_idx] = np.sum(y_nt * phase_shift)

    return [idx_info, s]

class postProcessing:
    def __init__(self, settings):
        self.name = settings["name"]  # 用于保存的文件名
        self.start_time = settings["start_time"]
        self.numFrames = settings["numFrames"]
        self.numADCSamples = settings["numADCSamples"]
        self.numTxAntennas = settings["numTxAntennas"]
        self.numRxAntennas = settings["numRxAntennas"]
        self.numVirtualAntennas = self.numTxAntennas * self.numRxAntennas
        self.numLoopsPerFrame = settings["numLoopsPerFrame"]
        self.startFreq = settings["startFreq"]
        self.sampleRate = settings["sampleRate"]
        self.freqSlope = settings["freqSlope"]
        self.idleTime = settings["idleTime"]
        self.rampEndTime = settings["rampEndTime"]
        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame
        self.numAngleBins = settings["numAngleBins"]
        self.waveLength = 0.005  # 天线间距配置
        self.progress = tqdm(total=9*17*9, desc="Beamforming")
        self.clip = settings["clip"]
        self.dataCube = self.load_data(settings["path"])

    def load_data(self, path):
        """
        读取数据
        ---
        返回值：(frames, chirps, 3tx, 4rx, 256 ADC - complex)
        """
        adc_data = np.array([])
        print("Loading file: ", path)
        adc_data = np.concatenate([adc_data, np.fromfile(path, dtype=np.int16)])
        self.numFrames = len(adc_data) // (2 * 2 * self.numADCSamples * self.numVirtualAntennas)
        #print("numFrames: ", self.numFrames)
        if self.clip == -1:
            self.clip = self.numFrames # 默认处理所有帧
        adc_data = adc_data.reshape(self.numFrames, -1)

        adc_data = np.reshape(
            adc_data,
            (
                -1,
                self.numLoopsPerFrame,
                self.numTxAntennas,
                self.numRxAntennas,
                self.numADCSamples // 2,
                2,
                2,
            ),
        )
        # Frames*Chirps*3TX*4RX*128(Samples//2)*2IQ(Lanes)*2Samples
        # print(adc_data.shape)

        adc_data = np.transpose(adc_data, (0, 1, 2, 3, 4, 6, 5))
        adc_data = np.reshape(
            adc_data,
            (
                -1,
                self.numLoopsPerFrame,
                self.numTxAntennas,
                self.numRxAntennas,
                self.numADCSamples,
                2,
            ),
        )
        # Frames*Chirps*3TX*4RX*256Samples*2IQ(Lanes)
        # print(adc_data.shape)

        """
        画iq图 已验证
        plt.figure()
        plt.plot(adc_data[0,1,2,0,:,0], 'b') # I
        plt.plot(adc_data[0,1,2,0,:,1], 'r') # Q
        plt.show()
        """

        adc_data = (
            1j * adc_data[:, :, :, :, :, 0] + adc_data[:, :, :, :, :, 1]
        ).astype(np.complex64)
        # Frames*Chirps*3TX*4RX*256Samples复数形式
        adc_data = adc_data[:self.clip, :, :, :, :] # 只取前clip帧
        #print(adc_data.shape)
        return adc_data

    def beamforming(self):
        # 体素网格尺寸和范围
        print("Beamforming start")
        voxel_grid_dimensions = (9, 9, 17)
        voxel_range = {
            "x": (-0.2, 0.2),
            "y": (0.35, 0.6),
            "z": (-0.25, 0.25),
        }
        self.voxel_range = voxel_range
        self.voxel_grid_dimensions = voxel_grid_dimensions
        voxel_step = {
            "x": (voxel_range["x"][1] - voxel_range["x"][0]) / (voxel_grid_dimensions[0] - 1),
            "y": (voxel_range["y"][1] - voxel_range["y"][0]) / (voxel_grid_dimensions[1] - 1),
            "z": (voxel_range["z"][1] - voxel_range["z"][0]) / (voxel_grid_dimensions[2] - 1),
        }
        # 创建三维体素网格
        x = np.linspace(voxel_range["x"][0], voxel_range["x"][1], voxel_grid_dimensions[0])
        y = np.linspace(voxel_range["y"][0], voxel_range["y"][1], voxel_grid_dimensions[1])
        z = np.linspace(voxel_range["z"][0], voxel_range["z"][1], voxel_grid_dimensions[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        voxel_positions = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)  # (n_voxels, 3)

        # 天线位置
        lamda = self.waveLength
        antenna_positions = np.array([
            [-lamda/4, 0, 5*lamda/4], [lamda/4, 0, 5*lamda/4],
            [-lamda/4, 0, 3*lamda/4], [lamda/4, 0, 3*lamda/4],  
            [-lamda/4, 0, lamda/4], [lamda/4, 0, lamda/4], [3*lamda/4, 0, lamda/4], [5*lamda/4, 0, lamda/4],
            [-lamda/4, 0, -lamda/4], [lamda/4, 0, -lamda/4], [3*lamda/4, 0, -lamda/4], [5*lamda/4, 0, -lamda/4]
        ])  # shape: (12, 3)

        # 使用Pool并行计算
        with Pool(processes=6) as pool:
            results = pool.starmap(calculation, [
                (self.dataCube, voxel_positions[i], (i // (voxel_grid_dimensions[1] * voxel_grid_dimensions[2]),
                                                    (i // voxel_grid_dimensions[2]) % voxel_grid_dimensions[1],
                                                    i % voxel_grid_dimensions[2]),
                 antenna_positions, {"numFrames": self.numFrames, "numRxAntennas": self.numRxAntennas,
                                     "freqSlope": self.freqSlope, "waveLength": self.waveLength, "clip": self.clip})
                for i in range(voxel_positions.shape[0])
            ])

        # 将结果汇总到S_xyz_t
        self.S_xyz_t = np.zeros((voxel_grid_dimensions[0], voxel_grid_dimensions[1], voxel_grid_dimensions[2], self.clip), dtype=np.complex64)
        for idx_info, s in results:
            self.S_xyz_t[idx_info[0], idx_info[1], idx_info[2], :] = s

        return self.S_xyz_t

    def preprocess_ecg(self, ecg_path):
        '''ecg_data处理'''
        # 读取ECG数据
        self.ecg_data = np.loadtxt(ecg_path, delimiter=',', skiprows=1, usecols=(0, 1))
        self.ecg_data[:, 0] = self.ecg_data[:, 0] / 1000  # 时间转换为秒
        self.ecg_data = self.ecg_data[:-255] # Remove last 255 rows from ecg_data
        # 此时 ecg_data 已经是一个 N×2 的 numpy 数组：
        # ecg_data[:,0] 是时间（秒），ecg_data[:,1] 是重采样后的 ECG 值
        # 把原来 256 Hz 的 ECG 重采样到 200 Hz 并覆盖 ecg_data
        # 原始采样率和目标采样率
        orig_fs = 256
        target_fs = 200

        # 原始样本点数
        old_N = self.ecg_data.shape[0]
        # 计算目标样本点数
        new_N = int(old_N * target_fs / orig_fs)

        # 提取原始时间和信号
        orig_time = self.ecg_data[:, 0].copy()
        orig_sig  = self.ecg_data[:, 1].copy()

        # 对信号列做重采样
        resampled_sig = resample(orig_sig, new_N)

        # 构造新的时间轴（等间隔）
        resampled_time = np.linspace(orig_time[0], orig_time[-1], new_N)

        # 用新的时间和信号覆盖 ecg_data
        self.ecg_data = np.column_stack((resampled_time, resampled_sig))

        # ecg clean
        self.ecg_data[:, 1] = nk.ecg_clean(self.ecg_data[:, 1], sampling_rate=200)

        # 归一化到[-500, 500]范围   
        old_min = np.min(self.ecg_data[:, 1])
        old_max = np.max(self.ecg_data[:, 1])
        self.ecg_data[:, 1] = -500 + (self.ecg_data[:, 1] - old_min) * (500 - -500) / (old_max - old_min)
        ecg_norm = self.ecg_data[:, 1]
        return ecg_norm

    def find_ecg_peaks(self, ecg_signal, height=100, distance=100):
        """
        检测 ECG 信号中的峰值位置。

        参数
        ----
        ecg_signal : 1D array-like
            输入的 ECG 振幅序列，比如 ecg_data[:,1]
        height : float or tuple, optional
            峰值的最小高度。若为单值，则所有峰值都需 >= height；
            若为 (min_height, max_height)，则峰值需在此范围内。
            默认 None，不作高度过滤。
        distance : int, optional
            相邻两个峰值之间的最小采样点数，用于去除过于接近的伪峰。
            默认 None，不作距离过滤。

        返回
        ----
        ecg_peaks : ndarray
            峰值在输入数组中的索引位置。
        properties : dict
            `find_peaks` 返回的其它属性字典，可选。

        """
        ecg_peaks, properties = find_peaks(ecg_signal, height=height, distance=distance)
        return ecg_peaks, properties


    def match_ecg(self, ecg_path, interp_method='linear'):
        """
        匹配ECG数据的时间戳，并补齐到36000步。
        参数：
            ecg_path: ECG数据文件路径。
            interp_method: 插值方法，默认是 'linear'。
        返回：
            补齐并截取后的ECG信号。
        """
        # 读取ECG数据
        self.ecg_data = np.loadtxt(ecg_path, delimiter=',', skiprows=1, usecols=(0, 1))
        self.ecg_data[:, 0] = self.ecg_data[:, 0] / 1000  # 时间转换为秒

        # 确保 start_time 是浮点数
        self.start_time = float(self.start_time.replace('\n', ''))

        # 计算起始时间
        self.ecg_start_time = self.ecg_data[0][0]
        self.common_start_time = max(self.ecg_start_time, self.start_time)

        # 固定结束时间为 common_start_time + 180 秒
        self.end_time = self.common_start_time + 180

        # 截取从 common_start_time 到 end_time 的 ECG 数据
        ecg_indices = np.where(
            (self.ecg_data[:, 0] >= self.common_start_time) &
            (self.ecg_data[:, 0] <= self.end_time)
        )
        self.ecg_data = self.ecg_data[ecg_indices]

        if len(self.ecg_data) == 0:
            raise ValueError("在指定的时间范围内未找到有效的ECG数据")

        # 创建新的时间序列
        time_serial = np.linspace(self.common_start_time, self.end_time, len(self.ecg_data))
        self.ecg_data[:, 0] = time_serial

        # 对ECG数据进行插值和补齐
        self.ecg_data = self.interpolate_ecg(interp_method=interp_method)

        #print('Resampled ECG data shape:', self.ecg_data.shape)

        # 毫米波数据的时间轴
        self.mmwave_time = np.linspace(0, 180, self.numFrames + 1)[:-1]

        # 截取毫米波数据的中间部分
        mid_point = self.mmwave_time.shape[0] // 2
        start_index = mid_point - 18000
        end_index = mid_point + 18000
        self.S_xyz_t = self.S_xyz_t[:, :, :, start_index:end_index]

        # 返回毫米波数据和补齐后的ECG信号
        return self.S_xyz_t, self.ecg_data




    # def save(self):
    #     """
    #     保存数据
    #     """
    #     with open("./pkl/{}.pkl".format((self.name)), "wb") as f:
    #         pickle.dump(
    #             [self.S_xyz_t, {
    #                 "voxel_range" : self.voxel_range,
    #                 "clip" : self.clip,
    #                 "voxel_grid_dimensions" : self.voxel_grid_dimensions,
    #                 "start_time": self.start_time
    #             }]
    #             , f)
    #     print("Save successfully!")
    def save(self):
        """
        保存数据
        """
        name = './pkl/matched_' + self.name + '.pkl'
        with open(name, 'wb') as f:
            pickle.dump([self.S_xyz_t, self.ecg_data], f)

    def callback(self, ret): # 多线程的回调函数
        self.S_xyz_t[ret[0][0], ret[0][1], ret[0][2], :] = ret[1]
        self.progress.update(1)

    

import pickle
import numpy as np
import matplotlib.pyplot as plt
from tslearn.metrics import dtw as tsl_dtw
from scipy.signal import resample
from librosa import zero_crossings
import time
from tqdm import tqdm
import neurokit2 as nk

def compute_second_derivative_vectorized(data, h):
    N = data.shape[-1]

    # 创建卷积核
    kernel = np.array([1, 2, -1, -4, -1, 2, 1]) / (16 * h**2)

    # 使用 np.pad 来处理边界条件
    padded_data = np.pad(data, ((0, 0), (0, 0), (0, 0), (3, 3)), mode='edge')

    # 使用 np.apply_along_axis 来应用卷积
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=-1, arr=padded_data)

def normalize_to_range(data, new_min, new_max):

    # Calculate the minimum and maximum of the original data
    min_val = np.min(data)
    max_val = np.max(data)

    # Apply the normalization formula
    normalized_data = new_min + ((data - min_val) / (max_val - min_val)) * (new_max - new_min)

    return normalized_data

class Focusing:
    def __init__(self, settings, radar_data, ecg_data):
        self.name = settings["name"]
        self.h_min = settings["h_min"]
        self.h_max = settings["h_max"]
        self.h = settings["h"]
        self.radar_data = radar_data
        self.ecg_data = ecg_data
        self.sequence = self.process_pkl()


    def process_pkl(self):
        #print('start processing')
        angles = np.unwrap(np.angle(self.radar_data), axis=-1, period = 0.05*np.pi)
        return compute_second_derivative_vectorized(angles, self.h)
    
    def focusing(self):
        s_list = []
        
        print('Focusing start')
        start_time = time.time()
        total_iterations = 9*9*17
        window_length = 4000
        with tqdm(total=total_iterations, desc="Processing") as pbar:
            for x in range(9):
                for y in range(9):
                    for z in range(17):
                        # (1) Divide the original motion signals into multiple non-overlap hmax length of segments as the coarse template candidates TC
                        start_index = np.random.randint(9000, self.sequence.shape[3] - window_length + 1 - 9000)
                        signal_xyz = self.sequence[x, y, z, start_index:start_index + window_length]
                        n_0crs = np.sum(zero_crossings(signal_xyz))

                        if n_0crs >= 1100:
                            s_last = np.inf
                            # if n_0crs > 1000 and n_0crs < 1100:
                            #     print(n_0crs)
                            #     fig, plot = plt.subplots(1, 1, figsize=(20, 2.5))
                            #     plot.plot(np.arange(0, 20, 0.005), signal_xyz[:], linewidth=0.5)
                            #     plot.set_xticks(np.arange(0, 20, 1))
                            #     plot.set_xlim(0, 20)
                            #     plot.grid()
                            #     plt.show()
                        else: 
                            signal_xyz = normalize_to_range(signal_xyz, -500, 500)
                            s_last = self.calculate_score(signal_xyz)
                        
                        if s_last != np.inf:
                            s_list.append([s_last, x, y, z])
                        pbar.update(1)

        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"Interations end, elapsed time: {elapsed_time:.2f} seconds")

        s_sorted = sorted(s_list, key=lambda x: x[0])
        # s_array = np.array(s_sorted[:200])
        s_array = np.array(s_sorted[:])
        #print('Number of points: {}'.format(s_array.shape[0]))

        signal_list = []
        points_list = []

        for i in range(len(s_array)):
            coordinate = s_array[i, 1:]
            x = int(coordinate[0])
            y = int(coordinate[1])
            z = int(coordinate[2])
            # print(self.sequence[x, y, z, :])
            signal_list.append(self.sequence[x, y, z, :])
            points_list.append(coordinate)
        final_list = self.save(signal_list, points_list)
        return final_list

    
    def calculate_score(self, signal_xyz):
        T_c = signal_xyz.reshape(-1,self.h_max)
        i_start = np.random.randint(T_c.shape[0])
        i = 0

        for i in range(4):
            if i == 0:
                T_dagger = T_c[i_start,:]
            else:
                T_dagger = self.template_upd(S_dagger)
            index, s_current = self.segments_upd(signal_xyz, T_dagger)

            if s_current > 31000:
                s_last = np.inf
            #     # print('n')
            #     # if s_current > 31000 and s_current < 35000:
            #     #     print(s_current)
            #     #     fig, plot = plt.subplots(1, 1, figsize=(20, 2.5))
            #     #     plot.plot(np.arange(0, 20, 0.005), signal_xyz[:], linewidth=0.5)
            #     #     plot.set_xticks(np.arange(0, 20, 1))
            #     #     plot.set_xlim(0, 20)
            #     #     plot.grid()

            #     #     plt.show()
                break
            elif s_current < 10000:
                s_last = 0
                # print('y')
                # fig, plot = plt.subplots(1, 1, figsize=(20, 2.5))
                # plot.plot(np.arange(0, 20, 0.005), signal_xyz[:], linewidth=0.5)
                # plot.set_xticks(np.arange(0, 20, 1))
                # plot.set_xlim(0, 20)
                # plot.grid()
                # plt.show()
                break

            S_dagger = self.non_overlapping_seg(signal_xyz, index)
            s_last = s_current

        return s_last

    def segments_upd(self, S, T_dagger):
        # (2) Rewrite the optimization problem into overlapping segmenting version and calculate the best overlapping matching segmentation
        s_T = []
        index = []
        i = 0
        index_current_T = []
        s_current_T = 0
        while i < (len(S) - self.h_max):
            s_temp = np.inf
            for n in range(self.h_min, self.h_max, 10):
                i_current = i+n
                s = tsl_dtw(T_dagger, S[i_current:(i_current+self.h_max)])
                if s < s_temp:
                    s_temp = s
                    n_temp = n
                    s_temp_T = s
            i += n_temp
            s_current_T += s_temp_T
            index_current_T.append(i)
        s_T.append(s_current_T)
        index.append(index_current_T)
        return index[np.argmin(s_T)], s_current_T
    
    def non_overlapping_seg(self, S, index):
        # (3) Reform the final non-overlapping segmentation
        non_overlapping_S = []
        for i in range(len(index) - 1):
            non_overlapping_S.append(S[index[i]:index[i+1]])
        return non_overlapping_S
    
    def template_upd(self, S_dagger):
        segment_sum = np.zeros(self.h_max)
        for segment in S_dagger:
            segment_itpl = resample(segment, self.h_max)
            segment_sum += segment_itpl
        T_dagger = segment_sum/len(S_dagger)
        return T_dagger


    def save(self, signal_list, points_list):
        """
        保存数据
        """
        signal_array = np.array(signal_list)
        points_array = np.array(points_list)
        ecg_array = nk.ecg_clean(self.ecg_data, sampling_rate=200)
        
        # fig, plot = plt.subplots(2, 1, figsize=(20, 5))
        # for i in range(len(points_array)):
        #     plot[0].plot(np.arange(0, 2.5, 0.005), signal_array[i,18000:18500], linewidth=0.5)
        #     plot[0].set_xticks(np.arange(0, 2.5, 1))
        #     plot[0].set_xlim(0, 2.5)
        #     plot[0].grid()

        #     if i <= 50:
        #         plot[1].plot(np.arange(0, 2.5, 0.005), signal_array[i,18000:18500], linewidth=0.5)
        #         plot[1].set_xticks(np.arange(0, 2.5, 1))
        #         plot[1].set_xlim(0, 2.5)
        #         plot[1].set_ylim(-800, 800)
        #         plot[1].grid()

        # ecg_segment = normalize_to_range(ecg_array[18000:18500], -600, 600)
        # plot[0].plot(np.arange(0, 2.5, 0.005), ecg_segment, color='red', linewidth=2)
        # plot[1].plot(np.arange(0, 2.5, 0.005), ecg_segment, color='red', linewidth=2)

        # plt.show()

        final_list = [signal_array, ecg_array, points_array]
        return final_list

from scipy.spatial.distance import cdist
class CardiacSignalClustering:
    def __init__(self, n_clusters, rho_s, rho_l, max_iter=100, tol=1e-6):
        self.n_clusters = n_clusters  # 簇的数量
        self.rho_s = rho_s  # 信号差异的权重
        self.rho_l = rho_l  # 位置差异的权重
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值

    def fit(self, S, L, P):
        """
        S: 形状为 (n_samples, n_timepoints) 的数组，表示运动信号
        L: 形状为 (n_samples, 3) 的数组，表示3D位置
        P: 形状为 (n_samples,) 的数组，表示信号功率
        D: 形状为 (n_samples,) 的数组，表示DTW距离
        """
        self.S = S
        self.L = L
        self.P = P
        self.n_samples, self.n_timepoints = S.shape

        # 初始化簇中心
        # 选择距离最小的 n_clusters 个样本作为初始中心
        idx = np.random.choice(self.n_samples, self.n_clusters, replace=False)
        self.mu = self.S[idx]
        self.l_mu = self.L[idx]

        for iteration in range(self.max_iter):
            old_mu = self.mu.copy()
            old_l_mu = self.l_mu.copy()

            # E步：分配样本到最近的簇
            self.labels = self._assign_clusters()

            # M步：更新簇中心
            self._update_centroids()

            # self._update_weights()

            # 检查收敛
            if self._check_convergence(old_mu, old_l_mu):
                #print(f"算法在第 {iteration + 1} 次迭代后收敛")
                break

        return self

    def _assign_clusters(self):
        """E步：将每个样本分配到最近的簇"""
        # 计算信号差异
        S_diff = cdist(
            self.S, self.mu, metric="euclidean"
        )  # 返回一个矩阵，其中 (i, j) 元素是 XA 中第 i 个点与 XB 中第 j 个点之间的距离
        # 计算位置差异
        L_diff = cdist(self.L, self.l_mu, metric="euclidean")

        # 计算总距离
        distances = self.rho_s * S_diff + self.rho_l * L_diff
        # 分配到最近的簇
        return np.argmin(distances, axis=1)  # 对应每个样本的最近簇的索引

    def _update_centroids(self):
        """M步：更新簇中心"""
        for k in range(self.n_clusters):  # 遍历每个簇（50）
            mask = self.labels == k  # 找到属于该簇的样本
            if np.sum(mask) > 0:
                weights = self.P[mask] / np.sum(self.P[mask])
                # print(weights)
                self.mu[k] = np.average(self.S[mask], axis=0, weights=weights)
                self.l_mu[k] = np.average(self.L[mask], axis=0, weights=weights).astype(
                    int
                )
            else:
                # 如果没有样本属于该簇，则随机选择一个样本作为中心
                idx = np.random.choice(self.n_samples)
                self.mu[k] = self.S[idx]
                self.l_mu[k] = self.L[idx]

    def _check_convergence(self, old_mu, old_l_mu):
        """检查算法是否收敛"""
        return (
            np.sum(np.abs(self.mu - old_mu)) < self.tol
            and np.sum(np.abs(self.l_mu - old_l_mu)) < self.tol
        )

    def get_cluster_centers(self):
        """返回簇中心"""
        return self.mu, self.l_mu



def sort_voxels_by_zero_crossings(rcg_data):
    """
    对形状为 (X, Y, Z, T) 的雷达数据，在最后一维上统计每个体素信号的过零点数量，
    并按从小到大排序体素坐标。

    参数
    ----
    rcg_data : ndarray, shape (X, Y, Z, T)
        雷达信号数据，前三个维度对应体素坐标，最后一个维度是时间序列。

    返回
    ----
    sorted_coords : ndarray, shape (X*Y*Z, 3)
        排序后的体素坐标列表，每一行是一个 (x, y, z) 三元组，
        顺序对应过零点计数从少到多。
    """
    # 计算符号
    signs = np.sign(rcg_data)
    # 过零点：相邻样本符号相乘为负
    zero_cross_counts = np.sum(signs[..., :-1] * signs[..., 1:] < 0, axis=-1)
    # 展平并 argsort
    flat_counts = zero_cross_counts.ravel()
    order = np.argsort(flat_counts)
    # 将 flat 索引转换回 (x, y, z)
    coords = np.array(np.unravel_index(order, zero_cross_counts.shape)).T
    return coords

def split_data_into_parts(data, part_length=11000, num_parts=3):
    """
    Split data into specified number of parts along the last dimension with specified length

    Parameters:
    -----------
    data : ndarray
        Input data array
    part_length : int
        Length of each part
    num_parts : int
        Number of parts to return

    Returns:
    --------
    list of ndarrays
        List containing the requested number of split data parts
    """
    parts = []
    for i in range(0, min(data.shape[-1], num_parts * part_length), part_length):
        part = data[..., i:i + part_length]
        if part.shape[-1] == part_length:  # Only include full-length parts
            parts.append(part)
        if len(parts) >= num_parts:  # Stop after getting requested number of parts
            break
    return parts[:num_parts]

def prune_excess_peaks(rcg_peaks, rcg_sample, bins=20, interval_range=None):
    """
    如果 rcg_peaks 数量超过阈值 len(rcg_sample) * 2.1/200，
    则删除那些与前一个峰间隔落在最常见区间内的峰。

    参数
    ----
    rcg_peaks : 1D array-like of int
        原始 RCG 峰值索引（升序）。
    rcg_sample : 1D array-like
        对应的 RCG 信号，用于计算阈值。
    bins : int or sequence, optional
        用于计算最常见峰间隔的直方图箱数或边界。默认 20。
    interval_range : tuple (min, max), optional
        直方图统计时限制的间隔范围，传给 np.histogram 的 range。

    返回
    ----
    pruned_peaks : ndarray of int
        修剪后的 RCG 峰值索引列表。
    """
    # 计算阈值：如果峰值数量超过这个就需要修剪
    threshold = len(rcg_sample) * 2.1 / 200
    rcg_peaks = np.asarray(rcg_peaks, dtype=int)
    if len(rcg_peaks) <= threshold:
        return rcg_peaks.copy()

    # 1. 找出最常见的峰间隔区间
    intervals = np.diff(rcg_peaks)
    counts, bin_edges = np.histogram(intervals, bins=bins, range=interval_range)
    idx = np.argmax(counts)
    bin_min, bin_max = bin_edges[idx], bin_edges[idx+1]

    # 2. 标记需要删除的峰：那些与前一个峰的间隔落在 [bin_min, bin_max) 内的峰
    #    注意：我们从索引 1 开始，因为第 0 个峰没有前驱
    to_remove = []
    for i in range(1, len(rcg_peaks)):
        if bin_min <= (rcg_peaks[i] - rcg_peaks[i-1]) <= bin_max:
            to_remove.append(i)

    # 3. 返回删除指定索引后的峰列表
    pruned_peaks = np.delete(rcg_peaks, to_remove)
    return pruned_peaks


def find_best_ecg_shift(ecg_peaks, rcg_peaks, tol=3, max_shift=None):
    """
    找到将 ecg_peaks 向左平移多少采样点，使得它们最贴近 rcg_peaks。

    参数
    ----
    ecg_peaks : array-like of int
        ECG 信号检测到的峰值索引列表（升序）。
    rcg_peaks : array-like of int
        RCG 信号检测到的峰值索引列表（升序）。
    tol : int, optional
        两个峰匹配时允许的索引误差范围（样本点）。默认 0。
    max_shift : int or None, optional
        搜索最大平移量。如果为 None，则自动设为
        max(ecg_peaks) - min(rcg_peaks)，保证不越界。

    返回
    ----
    best_shift : int
        最佳的向左平移量（样本点数）。
    best_count : int
        对齐成功（匹配）峰对的数量。
    """
    ecg = np.asarray(ecg_peaks, dtype=int)
    rcg = np.asarray(rcg_peaks, dtype=int)

    # 确定最大搜索平移
    if max_shift is None:
        max_shift = int(ecg.max() - rcg.max())
    if max_shift < 0:
        return 0, 0

    best_shift = 0
    best_count = 0

    # 对于每个可能的平移量 s，从 0 到 max_shift
    for s in range(0, max_shift + 1):
        # 将 ECG 峰向左平移 s
        shifted = ecg - s

        # 统计匹配个数
        count = 0
        # 对于每个平移后的位置，用二分查找判断是否有 RCG 峰在 tol 范围内
        for val in shifted:
            idx = np.searchsorted(rcg, val)
            if idx < len(rcg) and abs(rcg[idx] - val) <= tol:
                count += 1
            elif idx > 0 and abs(rcg[idx-1] - val) <= tol:
                count += 1

        # 更新最优
        if count > best_count:
            best_count = count
            best_shift = s

    return best_shift, best_count

def align_ecg_segments(rcg_parts, rcg_parts_peaks,
                       ecg_signal, ecg_peaks,
                       tol=0, max_shift=3000):
    """
    分段将 ECG 信号与多段 RCG 波形对齐。

    输入:
      - rcg_parts: list of 1D arrays, 每段 RCG 波形
      - rcg_parts_peaks: list of ndarray, 每段检测并修剪后的 RCG 峰索引
      - ecg_signal: 1D array, 原始（清洗+缩放后）ECG 信号
      - ecg_peaks: 1D array, ECG 信号的峰索引
      - tol: 对齐时峰值匹配的索引容差
      - max_shift: 最大平移量

    输出:
      - overlap_ecgs: list of 1D arrays, 每段对齐后与 RCG 等长的 ECG 片段
      - overlap_peaks_list: list of ndarray, 对应每段片段内的 ECG 峰索引
    """
    overlap_ecgs = []
    overlap_peaks_list = []

    # 初始残余 ECG
    res_ecg   = ecg_signal.copy()
    res_peaks = np.asarray(ecg_peaks, dtype=int)

    for sample, part_peaks in zip(rcg_parts, rcg_parts_peaks):
        L = sample.size

        # 找最佳平移
        shift, _ = find_best_ecg_shift(res_peaks, part_peaks,
                                       tol=tol, max_shift=max_shift)

        # 向左平移 ECG 并填 NaN
        M = len(res_ecg)
        shifted = np.full(M, np.nan)
        if shift < M:
            shifted[:M-shift] = res_ecg[shift:]

        # 平移峰值
        s_peaks = res_peaks - shift
        s_peaks = s_peaks[(s_peaks >= 0) & (s_peaks < M)]

        # 提取与当前 RCG 段重叠的 ECG 片段
        overlap_sig   = shifted[:L]
        overlap_peaks = s_peaks[s_peaks < L]

        overlap_ecgs.append(overlap_sig)
        overlap_peaks_list.append(overlap_peaks)

        # 更新残余 ECG 及峰值，继续对齐下一段
        res_ecg   = shifted[L:]
        res_peaks = s_peaks[s_peaks >= L] - L

    return overlap_ecgs, overlap_peaks_list











import os
from scipy.io import savemat


def save_segments_to_mat(rcg_parts_signals, overlap_ecgs, poz_xyz, csv_path, output_folder):
    """
    将多组 RCG、ECG 和同一组坐标保存为多个 .mat 文件。

    参数
    ----
    rcg_parts_signals : list of ndarray, 每个元素 shape (L, 50)
        多段 RCG 信号（已按 sorted_voxel_coords 前 50 求均值并转置）。
    overlap_ecgs : list of ndarray, 每个元素 shape (L,)
        对应每段的对齐后 ECG 信号片段。
    poz_xyz : ndarray, shape (50, 3)
        前 50 个体素的 (x,y,z) 坐标。
    csv_path : str
        原始 CSV 文件路径，用于生成文件基础名。
    output_folder : str
        保存 .mat 文件的目标文件夹。
    """
    # 基础文件名（去掉扩展名）
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # 逐段保存
    for idx, (rcg, ecg) in enumerate(zip(rcg_parts_signals, overlap_ecgs), start=1):
        # 构造带后缀的文件名
        file_name = f"{base_name}_{idx}.mat"
        save_path = os.path.join(output_folder, file_name)

        # mat 数据字典
        mat_data = {
            "data": [
                rcg,     # ['data'][0]: RCG (shape L×50)
                ecg,     # ['data'][1]: ECG (shape L,)
                poz_xyz  # ['data'][2]: pozXYZ (shape 50×3)
            ]
        }

        

        # 写文件
        savemat(save_path, mat_data)
        print(f"Saved segment {idx} to {save_path}")

def save_to_mat(rcg, ecg, poz_xyz, csv_path, output_folder):
    """
    将处理后的数据保存为 .mat 文件。
    """
    # 获取文件名（不含扩展名）
    file_name = os.path.splitext(os.path.basename(csv_path))[0] + ".mat"
    save_path = os.path.join(output_folder, file_name)

    # 创建数据字典
    mat_data = {
        "data": [
            rcg,    # ['data'][0] 是 RCG
            ecg,    # ['data'][1] 是 ECG
            poz_xyz # ['data'][2] 是 pozXYZ
        ]
    }

    # 保存为 .mat 文件
    savemat(save_path, mat_data)
    print(f"数据已成功保存为 {save_path}")

def process_root_folder(root_folder, part_length, num_parts):
    """
    遍历根文件夹，处理每个样本子文件夹中的文件，生成并保存 .mat 文件。
    """
    # 创建输出的 mat 文件夹
    output_folder = os.path.join(root_folder, "mat")
    os.makedirs(output_folder, exist_ok=True)

    # 遍历根文件夹中的每个子文件夹
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        if os.path.isdir(subdir_path):  # 确保是子文件夹
            # 找到子文件夹中的 txt, csv, bin 文件
            txt_file = None
            csv_file = None
            bin_file = None

            for file in os.listdir(subdir_path):
                if file.endswith(".txt") and "LogFile" not in file:
                    txt_file = os.path.join(subdir_path, file)
                elif file.endswith(".csv") and "LogFile" not in file:
                    csv_file = os.path.join(subdir_path, file)
                elif file.endswith(".bin"):
                    bin_file = os.path.join(subdir_path, file)
            
            # 检查是否找齐所需文件
            if txt_file and csv_file and bin_file:
                print(f"正在处理样本文件夹：{subdir}")
                
                try:# 调用算法处理
                    rcg_parts_signals, overlap_ecgs, poz_xyz_top50 = process(bin_file, txt_file, csv_file, root_folder, part_length, num_parts)


                    # 保存结果为 .mat 文件
                    save_segments_to_mat(rcg_parts_signals, overlap_ecgs, poz_xyz_top50, csv_file, output_folder)
                except Exception as e:
                    print(f"处理文件夹 {subdir} 时出错：{e}")
                    continue
            else:
                print(f"文件缺失，跳过文件夹：{subdir}")




def process(bin_path, log_path, ecg_path, root_folder, part_length, num_parts):
    # 从日志文件中读取开始时间
    with open(log_path, "r") as f:
        log = f.readlines()
        start_time = log[1].split(" ")[-1].strip().replace('\n', '')

    # 设置参数
    settings = {
    "path": os.path.normpath(bin_path),  # 规范化 bin_path
    "name": os.path.basename(bin_path).split("_")[0],
    "numFrames": 36000,
    "start_time": start_time,
    "clip": -1,
    "numADCSamples": 256,
    "numTxAntennas": 3,
    "numRxAntennas": 4,
    "numLoopsPerFrame": 2,
    "startFreq": 60,
    "sampleRate": 5000,
    "freqSlope": 64.997e12,
    "idleTime": 10,
    "rampEndTime": 60,
    "numAngleBins": 64,
    "h_min": 100,
    "h_max": 200,
    "h": 1 / 200,
}

    print(settings)

    # 创建处理器实例并运行
    processor = postProcessing(settings)
    rcg_data = processor.beamforming()
    angles = np.unwrap(np.angle(rcg_data), axis=-1, period = 0.05*np.pi)
    rcg_data = compute_second_derivative_vectorized(angles, 1 / 200)
    # 创建输出的 beamformed 文件夹
    beamformed_folder = os.path.join(root_folder, "beamformed")
    os.makedirs(beamformed_folder, exist_ok=True)
    # 保存 rcg_data 到 beamformed 文件夹
    beamformed_file = os.path.join(beamformed_folder, f"{os.path.basename(bin_path).split('_')[0]}_rcg_data.npy")
    np.save(beamformed_file, rcg_data)
    print(f"rcg_data 已保存到 {beamformed_file}")
    ecg_norm = processor.preprocess_ecg(ecg_path)
    print("ECG 信号预处理完成")
    ecg_peaks, props = processor.find_ecg_peaks(ecg_norm, height=100, distance=100)
    print("ECG 峰值检测完成")
    # rcg_data = np.load(r"C:\Users\21289364\OneDrive - LA TROBE UNIVERSITY\AirECG\AirECG_LabData\Test\test_pipeline_yijie\beamformed\1743575582.8525_rcg_data.npy")
    '''rcg_data排序和分割'''
    sorted_voxel_coords = sort_voxels_by_zero_crossings(rcg_data)
    # Split the data into parts
    rcg_parts = split_data_into_parts(rcg_data, part_length, num_parts)
    # 取过零点排序前五个体素坐标信号的平均值
    rcg_samples = []
    # Get signal from each part at the same voxel location
    for part in rcg_parts:
        # 取前 5 个体素坐标
        top5 = sorted_voxel_coords[:5]  # shape (5,3)
        # 提取这 5 个体素的信号，得到 shape (5, T)
        signals = part[top5[:,0], top5[:,1], top5[:,2], :]
        # 沿第 0 维求平均，得到 1×T 的平均信号
        avg_signal = np.mean(signals, axis=0)
        # 将平均信号加入列表
        rcg_samples.append(avg_signal)
    print("RCG 信号分段完成")
    '''对每段 RCG 波形做峰值检测并修剪'''
    rcg_parts_peaks = []
    for sample in rcg_samples:           # sample 是 1D 数组
        inv_sample = -sample           # 取反以检测负峰（原来正峰）
        # 峰值检测
        peaks, _ = find_peaks(
            inv_sample,
            height=(50, 600),
            distance=50,
            width=(0, 200),
            rel_height=1,
            prominence=(100, 1000),
            wlen=20
        )
        # 修剪多余的峰
        pruned = prune_excess_peaks(peaks, sample, bins=20, interval_range=(50, 100))
        rcg_parts_peaks.append(pruned)
    print("RCG 峰值检测完成")
    '''将切割好的 RCG 逐段与 ECG 对齐'''
    overlap_ecgs, overlap_peaks_list = align_ecg_segments(
        rcg_samples,
        rcg_parts_peaks,
        ecg_norm,
        ecg_peaks,
        tol=3,
        max_shift=3000
    )
    print("RCG 和 ECG 对齐完成")
    '''筛选rcg的前50个体素点信号'''
    top50 = sorted_voxel_coords[:50].astype(int)
    # 用于存放每个分段对应的 50 条信号
    # 最终 rcg_parts_signals[i] 形状为 (50, L)
    rcg_parts_signals = []

    for part in rcg_parts:
        # part.shape == (9, 9, 17, L)
        # 提取 50 条信号
        signals = np.array([
            part[x, y, z, :] 
            for (x, y, z) in top50
        ])  # shape -> (50, L)
        rcg_parts_signals.append(signals.T) # 转置后，每个元素 shape == (L, 50)
    print("RCG 信号筛选50个体素完成")

    return rcg_parts_signals, overlap_ecgs, top50
    
def main(path, part_length, num_parts):  # 根文件夹
    process_root_folder(path, part_length, num_parts)



