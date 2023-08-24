import os
import numpy as np
import pandas as pd
import time
from datetime import datetime
class DataFeatureMerge(object):
    def __init__(self):
        # 采集数据csv保存路径
        self.data_path = './data_set/structure_data/'
        # 图像特征数据csv保存路径
        self.feature_path = './data_set/extracted_features/'
        # 合并后所有数据的保存路径
        self.merged_data_path = './data_set/merged_data/'

    def get_data_info(self, batch_code):
        """
        读取指定批次的数据
        :param batch_code:
        :return: 采集数据csv，每个特征记录的上传时间，格式为"YYmmddHHMMSS"的int形式
        """
        path = self.data_path + batch_code + "-data.csv"
        data_file = pd.read_csv(path).copy()
        # date转换为"YYmmddHHMMSS"的int形式，与图片名称后半段匹配
        data_time = data_file.iloc[:, [2]]
        data_time = np.array(data_time).flatten()
        converted_time = []
        for i in range(len(data_time)):
            struct_time = time.strptime(data_time[i], "%Y-%m-%d %H:%M:%S")
            int_time = int(str(time.strftime("%Y%m%d%H%M%S", struct_time))[2:])
            converted_time.append(int_time)
        return data_file, converted_time

    def get_feature_info(self, batch_code):
        """
        读取指定批次的图像特征值
        :param batch_code:
        :return: 特征csv，每个特征记录的上传时间，格式为"YYmmddHHMMSS"的int形式
        """
        path = self.feature_path + batch_code + "-feat.csv"
        feature_file = pd.read_csv(path).copy()
        # 转换为int方便求时间差
        feature_time = np.array(feature_file.iloc[:, [3]]).flatten().tolist()
        return feature_file, feature_time

    def calculate_delta_time(self, str1, str2):
        """
        计算两个时间的时间差
        :param str1: "YYYY-mm-dd HH:MM:SS"形式的字符串
        :param str2: "YYYY-mm-dd HH:MM:SS"形式的字符串
        :return: str1和str2时间差的绝对值，单位 min
        """
        t1 = datetime.strptime(str1, "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(str2, "%Y-%m-%d %H:%M:%S")
        delta_t = 0
        if t1 > t2:
            delta_t = (t1 - t2).seconds
        else:
            delta_t = (t2 - t1).seconds
        delta_t = delta_t / 60.0
        return delta_t

    def merge_one_batch(self, batch_code):
        """
        合并、保存一个指定批次的采集数据及图像特征
        """
        data_file, data_time = self.get_data_info(batch_code)
        feature_file, feature_time = self.get_feature_info(batch_code)
        # 辅助变量
        fail_idx_list = []  # merge失败的feature row记录
        is_first_data = True  # 初始重量标志位
        original_weight = 0
        delta_time_threshold = 450  # 认为是同一组数据的时间差阈值
        # 后来加的“几成黄”列，添在最后一列
        yellow_col = feature_file['ye_per']
        feature_file.drop(labels=['ye_per'], axis=1, inplace=True)
        # 为每组feature匹配data（将匹配的data合并到feature中），通常len(data)>len(feature)
        for feat_idx in range(len(feature_time)):
            for data_idx in range(len(data_time)):
                delta_time = data_time[data_idx] - feature_time[feat_idx]
                # 采集数据和图片上传时间相差约±5min以内即认为是同一组数据
                if abs(delta_time) <= delta_time_threshold:
                    feature_file.loc[feat_idx, "dry_up"] = data_file.iat[data_idx, 4]
                    feature_file.loc[feat_idx, "dry_down"] = data_file.iat[data_idx, 5]
                    feature_file.loc[feat_idx, "wet_up"] = data_file.iat[data_idx, 6]
                    feature_file.loc[feat_idx, "wet_down"] = data_file.iat[data_idx, 7]
                    curr_weight = data_file.iat[data_idx, 8]
                    feature_file.loc[feat_idx, "weight"] = curr_weight
                    # 记录初始重量
                    if is_first_data:
                        original_weight = data_file.iat[data_idx, 8]
                    # 已烘烤时间
                    feature_file.loc[feat_idx, "baketime"] = data_file.iat[data_idx, 12]
                    driage = data_file.iat[data_idx, 10]
                    if np.isnan(driage):
                        driage = round(np.true_divide((original_weight - curr_weight), original_weight), 5)
                    feature_file.loc[feat_idx, "driage"] = driage
                    # 计算失水速率(单位: g/min)，从第2组数据开始计算
                    waterloss_rate = 0
                    if not is_first_data:
                        prev_weight = feature_file.iat[feat_idx - 1, 35]
                        spent_time = self.calculate_delta_time(data_file.iat[data_idx, 2],
                                                               feature_file.iat[feat_idx - 1, 39])
                        # 存在图片上传间隔<5min导致相邻2张图匹配到同一组weight的情况
                        if spent_time == 0:
                            spent_time = 1.0
                        waterloss_rate = round(np.true_divide(prev_weight - curr_weight, spent_time), 5)
                    feature_file.loc[feat_idx, "waterloss_rate"] = waterloss_rate
                    # 顺带记录data的upload_time
                    feature_file.loc[feat_idx, "data_time_format"] = data_file.iat[data_idx, 2]
                    feature_file.loc[feat_idx, "data_time"] = data_time[data_idx]
                    # 后来加的“状态标注”“稳温时间”列，添在最后一列
                    feature_file.loc[feat_idx, "label"] = data_file.iat[data_idx, 11]
                    feature_file.loc[feat_idx, "holdingtime"] = data_file.iat[data_idx, 13]
                    is_first_data = False
                    break
                # 剪枝：datalist按uploadtime升序
                if delta_time > delta_time_threshold:
                    fail_idx_list.append(feat_idx)
                    print("CANNOT match batch: {0}, upload time: {1}, failure count: {2}".format(batch_code,
                                                                                                 feature_time[feat_idx],
                                                                                                 len(fail_idx_list)))
                    break
        # 删除无匹配data的feature rows
        feature_file.drop(labels=fail_idx_list, axis=0, inplace=True)
        # 保存结果
        feature_file['yel_per'] = yellow_col
        feature_file.drop(labels=["Unnamed: 0"], axis=1, inplace=True)
        save_path = self.merged_data_path + batch_code + ".csv"
        feature_file.to_csv(save_path, index=True)
        return True

    def calculate_yellow_proportion(self, image):
        """
        基于hsv颜色阈值的“几成黄”计算
        :param image: hsv image
        :return: 图像的中黄橙色所占比例
        """
        yellow_area = self.hsv_generate_mask(image, 11, 34, 43, 255, 35, 255)  # 橙色+黄色
        yellow_pixCnt = len(yellow_area[yellow_area == 255])
        roi_pixCnt = self.count_not_black_pixels(image)
        proportion = round(np.true_divide(yellow_pixCnt, roi_pixCnt), 4)
        return proportion

def batch_mergedData_combine(src_path):
    filename_list = os.listdir(src_path)
    DF_arr = []
    for filename in filename_list:
        file_path = os.path.join(src_path, filename)
        csvfile = pd.read_csv(file_path).copy()
        DF_arr.append(csvfile)
    all_data = pd.concat(DF_arr, ignore_index=True)
    all_data.drop(labels=["Unnamed: 0"], axis=1, inplace=True)
    all_data.to_csv('./all_datas.csv', index=True)


if __name__ == '__main__':
    dfm = DataFeatureMerge()
    # 图片批次的存储路径
    base_path = './data_set/batch_images/'
    batch_list = os.listdir(base_path)
    cnt, total = 0, str(len(batch_list))
    for batch_code in batch_list:
        isSucceed = dfm.merge_one_batch(batch_code)
        if isSucceed:
            cnt = cnt + 1
            print("Merge success: " + batch_code + ", finished: " + str(cnt) + "/" + total)
