'''
采完雷达和ecg数据后的后续处理，包含以下几个步骤：
1. 解析ecg数据。将ecg数据解析为csv文件，每个csv文件包含一个ecg数据段。
2. 移动雷达和ecg数据移动到Dataset文件夹下。
'''

import os
import shutil
from datetime import datetime
import re
import csv
import os

################################################################################
# 以下为解析 NetAssist 数据的代码，将数据提取为 CSV 文件
def extract_timestamps(text, headers):
    """
    对于每个固定头部，从其结束位置开始搜索第一个方括号中的内容，
    将其作为该数据段的时间戳（不做任何修改）。若未找到，则返回 None。
    """
    timestamps = []
    for header in headers:
        start_index = header.end()
        ts_match = re.search(r'\[([^\]]+)\]', text[start_index:])
        if ts_match:
            timestamps.append(ts_match.group(1))
        else:
            timestamps.append(None)
    return timestamps

def extract_segment_with_replacement(text, start_index, target_length=768):
    """
    从 text 的 start_index 开始逐字符提取，累计 target_length 个字符。
    在提取过程中，如果遇到形如 "[1742276876.2026932]\n" 这样的模式，
    则将整个匹配替换为一个空格（只计入一个字符），然后继续提取，
    直至累计 target_length 个字符。
    """
    result = []
    index = start_index
    pattern = re.compile(r'\[[0-9\.]+\]\n')
    
    while len(result) < target_length and index < len(text):
        if text[index] == '[':
            m = pattern.match(text, index)
            if m:
                result.append(' ')
                index += m.end() - m.start()
                continue
        result.append(text[index])
        index += 1
    return ''.join(result)

def extract_data_segments(file_path):
    """
    从指定 txt 文件中提取数据段及对应的时间戳：
      1. 在整个文本中查找所有固定头部 "DB 03 01 46 DA FE 14 00" 的位置。
      2. 对于每个固定头部，从其结束位置开始查找第一个方括号内的内容，
         作为时间戳（不做任何修改）。
      3. 重新查找所有固定头部，从其结束位置开始提取 768 个字符作为数据段，
         在提取过程中遇到形如 "[1742276876.2026932]\n" 的模式时，替换为一个空格（只计入一个字符）。
    返回：
      data_segments: 列表，每个元素为一个元组 (timestamp, segment)
                     timestamp 为对应的时间戳（字符串，若未找到则为 None），
                     segment 为处理后的长度为 768 的字符串。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 查找所有固定头部的位置
    header_pattern = re.compile(r'DB 03 01 46 DA FE 14 00')
    headers = list(header_pattern.finditer(text))
    print(f"找到固定头部数量: {len(headers)}")
    
    # 提取时间戳：从每个固定头部后查找第一个方括号内的内容
    timestamps = extract_timestamps(text, headers)
    
    # 重新利用所有固定头部提取数据段（数据段紧跟固定头部，不受时间戳影响）
    segments = []
    for header in headers:
        start_index = header.end()
        segment = extract_segment_with_replacement(text, start_index, 768)
        if len(segment) == 768:
            segments.append(segment)
        else:
            print("提取到的数据段长度不是768个字符，跳过该数据段。")
    
    if not (len(headers) == len(timestamps) == len(segments)):
        print(f"警告：固定头部数量({len(headers)})、时间戳数量({len(timestamps)})与数据段数量({len(segments)})不一致！")
    
    return list(zip(timestamps, segments))

def write_csv(data_segments, output_csv_file):
    """
    输入 data_segments（列表，每个元素为 (timestamp, segment)），输出一个两列的 CSV 文件：
      第一列 'timestamp'：将每个提取到的时间戳复制 256 次，再按顺序排列成一列，
                           对时间戳进行处理，保留三位小数后去掉小数点；
      第二列 'Channel 1'：将所有数据段组合到一起，按空格分割后提取所有2位16进制数，
                           转换为10进制数，按顺序排列成一列。
    """
    all_timestamps = []
    all_channel1 = []
    
    for timestamp, segment in data_segments:
        # 对 timestamp 进行格式化：如果不为 None，则转换为浮点数，
        # 保留三位小数后去掉小数点；否则为空字符串
        if timestamp is not None:
            try:
                ts_float = float(timestamp)
                # 格式化为三位小数的字符串，再移除小数点
                ts_formatted = f"{ts_float:.3f}".replace(".", "")
                ts = ts_formatted
            except ValueError:
                ts = timestamp
        else:
            ts = ""
        
        # 将数据段按空格分割，得到2位16进制数列表（期望每个数据段有256个数字）
        tokens = segment.split()
        if len(tokens) != 256:
            print(f"警告：数据段的 hex 数量不是256, 实际数量: {len(tokens)}")
        try:
            decimal_values = [int(token, 16) for token in tokens]
        except ValueError as e:
            print("转换16进制为10进制时出错，请检查数据格式：", e)
            decimal_values = []
        # 将该数据段的时间戳复制为与数字数量相同（一般为256）
        all_timestamps.extend([ts] * len(decimal_values))
        all_channel1.extend(decimal_values)
    
    # 将两列数据写入 CSV 文件
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'Channel 1'])
        for ts, ch in zip(all_timestamps, all_channel1):
            writer.writerow([ts, ch])
    
    print(f"CSV 文件已保存到 {output_csv_file}")

def process_folder(folder_path):
    """
    输入文件夹路径，将该文件夹内所有 .txt 文件依次解析，
    为每个文件生成一个 CSV 文件（输出路径与输入路径相同，文件名相同，仅扩展名不同）。
    """
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            print(f"正在处理文件: {file_path}")
            data_segments = extract_data_segments(file_path)
            output_csv_file = os.path.splitext(file_path)[0] + ".csv"
            write_csv(data_segments, output_csv_file)
    print("所有文件处理完毕。")
################################################################################


def group_radar_data(source_dir):
    """
    1. 分组雷达数据：
       - 遍历 source_dir 下所有文件（不包括子文件夹）。
       - 按文件名前 13 个字符分组（不足 13 字符则以整个文件名分组）。
       - 按分组键升序排序。
       - 在当前工作目录下创建以当天日期（YYYYMMDD）命名的文件夹，
         并在其中依次创建子文件夹（1, 2, 3, ...）存放对应组的文件。
    返回：创建的日期文件夹路径。
    """
    today_str = datetime.now().strftime("%Y%m%d")
    date_folder = os.path.join(os.getcwd(), today_str)
    os.makedirs(date_folder, exist_ok=True)

    # 获取所有文件（不包含子目录）
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # 按文件名前13个字符分组
    groups = {}
    for f in files:
        key = f if len(f) < 13 else f[:13]
        groups.setdefault(key, []).append(f)
    
    # 对分组键进行升序排序
    sorted_keys = sorted(groups.keys())
    
    # 为每个分组创建一个子文件夹，并移动对应的文件
    for index, key in enumerate(sorted_keys, start=1):
        group_folder = os.path.join(date_folder, str(index))
        os.makedirs(group_folder, exist_ok=True)
        for filename in groups[key]:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(group_folder, filename)
            shutil.move(src, dst)
    
    return date_folder

def move_csv_data(netassist_dir, date_folder):
    """
    2. 移动 CSV 数据：
       - 从 netassist_dir 中获取所有 CSV 文件（文件名不区分大小写），
         并按文件名中对应的数字由小到大排序（避免字符串排序带来的顺序问题）。
       - 按排序顺序将 CSV 文件分别移动到 date_folder 下对应的子文件夹中（1, 2, 3, ...）。
         如果 CSV 数量超过子文件夹数量，则只处理与子文件夹对应数量的 CSV 文件。
    """
    # 获取所有 CSV 文件
    csv_files = [f for f in os.listdir(netassist_dir)
                 if os.path.isfile(os.path.join(netassist_dir, f)) and f.lower().endswith(".csv")]
    
    # 对 CSV 文件按文件名对应的数字排序
    # 假定文件名格式为 "数字.csv"，例如 "1.csv", "22.csv" 等
    csv_files_sorted = sorted(csv_files, key=lambda f: int(os.path.splitext(f)[0]))
    
    # 将 CSV 文件依次移动到对应的组文件夹中
    for i, csv_filename in enumerate(csv_files_sorted, start=1):
        group_folder = os.path.join(date_folder, str(i))
        if os.path.exists(group_folder):
            src_csv = os.path.join(netassist_dir, csv_filename)
            dst_csv = os.path.join(group_folder, csv_filename)
            shutil.move(src_csv, dst_csv)
        else:
            break  # 无对应组文件夹时停止处理

def move_txt_data(netassist_dir):
    """
    3. 移动 TXT 数据：
       - 在 netassist_dir 下的 data_recieved 文件夹中创建一个以当天日期（YYYYMMDD）命名的文件夹。
       - 将 netassist_dir 下所有 TXT 文件（不包含子目录）移动到该新建文件夹中。
    返回：TXT 文件目标文件夹的路径。
    """
    today_str = datetime.now().strftime("%Y%m%d")
    data_received_dir = os.path.join(netassist_dir, "data_recieved")
    os.makedirs(data_received_dir, exist_ok=True)
    
    txt_target_folder = os.path.join(data_received_dir, today_str)
    os.makedirs(txt_target_folder, exist_ok=True)
    
    txt_files = [f for f in os.listdir(netassist_dir)
                 if os.path.isfile(os.path.join(netassist_dir, f)) and f.lower().endswith(".txt")]
    
    for txt_filename in txt_files:
        src_txt = os.path.join(netassist_dir, txt_filename)
        dst_txt = os.path.join(txt_target_folder, txt_filename)
        shutil.move(src_txt, dst_txt)
    
    return txt_target_folder

def postcollectprocess(source_dir, netassist_dir, destination_base):

    # 1. 分组雷达数据，并获取创建的日期文件夹路径
    date_folder = group_radar_data(source_dir)

    # 2. 解析ecg源数据成csv文件
    process_folder(netassist_dir)
    
    # 3. 移动 CSV 数据到对应组文件夹中
    move_csv_data(netassist_dir, date_folder)
    
    # 4. 移动 TXT 数据到 netAssist 的 data_recieved 目录下
    txt_target_folder = move_txt_data(netassist_dir)
    
    # 将日期文件夹（包含雷达数据及 CSV 文件的子文件夹）移动到目标目录中
    today_str = os.path.basename(date_folder)
    final_destination = os.path.join(destination_base, today_str)
    shutil.move(date_folder, final_destination)
    
    print(f"雷达数据和 CSV 文件已成功移动到: {final_destination}")
    print(f"TXT 文件已成功移动到: {txt_target_folder}")

    return final_destination, txt_target_folder

if __name__ == '__main__':
    # 路径配置
    source_dir = r"C:\Users\Lab\Desktop\airecg\radar_data"
    netassist_dir = r"C:\Users\Lab\OneDrive - LA TROBE UNIVERSITY\AirECG\AirECG_LabData\netAssist-0.9"
    destination_base = r"C:\Users\Lab\OneDrive - LA TROBE UNIVERSITY\AirECG\AirECG_LabData\Dataset"
    dataset_folder, raw_ecg_folder = postcollectprocess(source_dir, netassist_dir, destination_base)
