"""
计算图像中的颜色和纹理特征值
"""
import os
import cv2
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

class ExtractFeature(object):
    def __init__(self, processed_file_path, feature_data_path):
        """
        :param self.processed_file_path: 处理图片储存的总路径
        :param self.feature_data_path: 特征数据存放位置
        """
        self.processed_file_path = processed_file_path
        self.feature_data_path = feature_data_path


    # 计算rgb值
    def extract_rgb(self, img, file_name):
        """
        :return: 图片无法打开返回空，正常打开返回datetime，rgb数据。
        """
        # 批次
        dt = [file_name[:12]]
        # 计算rgb
        b, g, r = self.filter_black(img)
        # 归一化
        b_normalization = b / (b + g + r)
        g_normalization = g / (b + g + r)
        r_normalization = r / (b + g + r)
        dt.extend([b_normalization, g_normalization, r_normalization])
        return dt

    # 去除黑色底色rgb值
    @staticmethod
    def filter_black(img):
        count = 0
        sum_r = 0
        sum_g = 0
        sum_b = 0
        for h in range(img.shape[0]):
            for w in range(img.shape[1]):
                if img[h][w][0] != 0 and img[h][w][1] != 0 and img[h][w][2] != 0:
                    count += 1
                    sum_b += img[h][w][0]
                    sum_g += img[h][w][1]
                    sum_r += img[h][w][2]
        return round(sum_b / count), round(sum_g / count), round(sum_r / count)

    # 计算hsv
    @staticmethod
    def extract_hsv(img):
        """
        计算图片的hsv颜色空间值
        """
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_h, img_s, img_v = cv2.split(img_hsv)
        H = img_h.astype(np.float64) * 2
        S = img_s.astype(np.float64) / 255
        V = img_v.astype(np.float64) / 255
        h_mean = np.mean(H)
        s_mean = np.mean(S)
        v_mean = np.mean(V)
        return h_mean, s_mean, v_mean

    # 计算Lab
    @staticmethod
    def extract_Lab(r, g, b):
        r = r / 255.0  # rgb range: 0 ~ 1
        g = g / 255.0
        b = b / 255.0

        if r > 0.04045:
            r = pow((r + 0.055) / 1.055, 2.4)
        else:
            r = r / 12.92

        if g > 0.04045:
            g = pow((g + 0.055) / 1.055, 2.4)
        else:
            g = g / 12.92

        if b > 0.04045:
            b = pow((b + 0.055) / 1.055, 2.4)
        else:
            b = b / 12.92

        X = r * 0.436052025 + g * 0.385081593 + b * 0.143087414
        Y = r * 0.222491598 + g * 0.716886060 + b * 0.060621486
        Z = r * 0.013929122 + g * 0.097097002 + b * 0.714185470

        X = X * 100.000
        Y = Y * 100.000
        Z = Z * 100.000

        X = X / 96.4221
        Y = Y / 100.000
        Z = Z / 82.5211

        if X > 0.008856:
            X = pow(X, 1 / 3.000)
        else:
            X = (7.787 * X) + (16 / 116.000)

        if Y > 0.008856:
            Y = pow(Y, 1 / 3.000)
        else:
            Y = (7.787 * Y) + (16 / 116.000)

        if Z > 0.008856:
            Z = pow(Z, 1 / 3.000)
        else:
            Z = (7.787 * Z) + (16 / 116.000)

        Lab_L = round((116.000 * Y) - 16.000, 2)
        Lab_a = round(500.000 * (X - Y), 2)
        Lab_b = round(200.000 * (Y - Z), 2)
        Lab = [Lab_L, Lab_a, Lab_b]
        return Lab

    # 计算颜色矩，包括一阶矩 二阶矩 三阶矩
    @staticmethod
    def extract_moments(img):
        # b均值、b方差、b偏移量；g均值、g方差、g偏移量；r均值、r方差、r偏移量
        mu = np.mean(img[:, :, 1])  # 求均值
        b_delta = np.std(img[:, :, 0])  # 求方差
        g_delta = np.std(img[:, :, 1])  # 求方差
        skew = np.mean(stats.skew(img[:, :, 1]))  # 求偏移量
        moments = [mu, b_delta, g_delta, skew]
        return moments

    def extract_feature_for_batch(self):
        """
        计算批次内图像特征值
        """
        print("特征计算中...")
        batch_paths = os.listdir(self.processed_file_path)
        for batch_path_name in tqdm(batch_paths):
            data = pd.DataFrame()
            info = []
            batch_path = self.processed_file_path + '/' + batch_path_name + '/'
            file_names = os.listdir(batch_path)
            for file_name in tqdm(file_names):
                print(batch_path + file_name)
                img = cv2.imread(batch_path + file_name)
                # 有些损坏的图片无法打开，直接删去
                if img is None:
                    print("PIC ERROR, DELETE：" + file_name)
                    real_file_path = batch_path + file_name
                    os.remove(real_file_path)
                    return
                rgb = self.extract_rgb(img, file_name)
                hsv = self.extract_hsi(img)
                Lab = self.extract_Lab(rgb[4], rgb[3], rgb[2])
                rgb.extend(hsv)
                rgb.extend(Lab)
                info.append(rgb)
            data = data.append(info, ignore_index=True)
            data.columns = ['batch','B', 'G', 'R',
                            'H', 'S', 'V','L', 'a', 'b',
                            ]

            print(data)
            data_saved = self.feature_data_path +"/" +batch_path_name + ".csv"
            data.to_csv(data_saved, index=True)
            print("特征储存完成...")


if __name__ == '__main__':
    # 图片读取路径，该文件夹下有若干批次
    processed_file = r'F:\pythoncode\Pic_file\Seg_Progressed_file'
    # 数据存放位置，该文件夹下会自动生成若干个存有数据的xls文件
    feature_data = r"F:\pythoncode\Date_file"
    CF = ExtractFeature(processed_file, feature_data)
    CF.extract_feature_for_batch()
