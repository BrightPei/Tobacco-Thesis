"""
对数据进行清洗，并且进行平滑滤波处理
"""
from scipy import signal
import numpy as np
class DataCleaning(object):

    def ordinal_encoder(data):
        data.loc[data["label"] == 'Initial yellowing', 'label'] = 1
        data.loc[data["label"] == 'Yellowing', 'label'] = 2
        data.loc[data["label"] == 'Yellowing and withering', 'label'] = 3
        data.loc[data["label"] == 'Tendon changing', 'label'] = 4
        data.loc[data["label"] == 'Dry flake', 'label'] = 5
        data.loc[data["label"] == 'Dry tendon', 'label'] = 6
        data['label'] = data['label'].astype(np.int64)
        return data

    def feature_normalize(data, features):
        """
        z-score标准化
        """
        data_mean = data[features].mean()
        data_std = data[features].std()
        data.loc[:, features] = (data[features] - data_mean) / data_std
    def detect_outliers(self, col):
        """
        箱型图异常识别，四分位法
        """
        QL = col.quantile(0.25)
        QU = col.quantile(0.75)
        IQR = QU - QL
        outlier = col[(col < QL - IQR * 1.5) | (col > QU + 1.5 * IQR)]
        res = outlier.index.values
        return res

    def fix_outliers(self, col, outlier_idx):
        """
        异常值处理：用异常点的前后采样点求平均值来填充异常值
        """
        fix_col = col.copy()
        for idx in outlier_idx:
            # 边界处理
            if idx == 0:
                fix_col.iloc[idx] = fix_col.iloc[1]
                continue
            if idx == len(fix_col) - 1:
                fix_col.iloc[idx] = fix_col.iloc[len(fix_col) - 2]
                continue
            fix_col.iloc[idx] = (fix_col.iloc[idx - 1] + fix_col.iloc[idx + 1]) / 2.0
        return fix_col

    def origin_data_process(self, data):
        """
        原始数据的清洗（异常值、缺失值）&滤波平滑处理
        """
        # 异常值处理：目前只有weight1有异常数据
        for ci in range(7, 8):
            outlier_idx_arr = self.detect_outliers(data.iloc[:, ci])
            if len(outlier_idx_arr) > 0:
                data.iloc[:, ci] = self.fix_outliers(data.iloc[:, ci], outlier_idx_arr)
        # 重量单位 kg -> g
        data['WEIGHT1'] = data['WEIGHT1'] * 1000
        # Savitzky-Golay Filter
        filter_param = [15, 1]
        data['DRY_BULB_T1'] = signal.savgol_filter(data['DRY_BULB_T1'], filter_param[0], filter_param[1])
        data['DRY_BULB_T2'] = signal.savgol_filter(data['DRY_BULB_T2'], filter_param[0], filter_param[1])
        data['WET_BULB_T1'] = signal.savgol_filter(data['WET_BULB_T1'], filter_param[0], filter_param[1])
        data['WET_BULB_T2'] = signal.savgol_filter(data['WET_BULB_T2'], filter_param[0], filter_param[1])
        data['WEIGHT1'] = signal.savgol_filter(data['WEIGHT1'], 5, 1)