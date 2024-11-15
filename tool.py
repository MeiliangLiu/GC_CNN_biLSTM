import csv
import numpy as np


def dream_read_label(file_path, p):
    label = np.zeros((p, p))
    with open(file_path, "r") as file:
        # 逐行读取文件内容
        for line in file:
            lines = line.strip().split("\t")  # 去除行尾的换行符和空格
            # print(lines)
            label[int(lines[1].replace("G", "")) - 1][int(lines[0].replace("G", "")) - 1] = lines[2]
    return label


def dream_read_data(file_path):
    data = []
    batch = []
    # 打开文件
    with open(file_path, "r") as file:
        # 创建 CSV 读取器，设置分隔符为制表符
        reader = csv.reader(file, delimiter="\t")
        # 逐行读取文件内容
        for i, row in enumerate(reader):
            if i == 0:
                continue
            # 文件中的21行通常是空行
            if i % 22 == 0:
                data.append(batch)
                batch = []
            else:
                # print(row)
                float_list = [float(x) for x in row[1:]]
                batch.append(float_list)
        data.append(batch)
    return np.array(data)
