from scipy.signal import find_peaks
import numpy as np


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