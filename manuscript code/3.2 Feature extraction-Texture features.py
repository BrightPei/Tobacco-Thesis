"""
提取图像中的纹理特征值
"""
import cv2
import os
import pandas as pd
import skimage
from tqdm import tqdm

def compute_glcm(img):
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    img_gray = img_gray[200:750, 600:1600]
    # 共生矩阵为四维，前两维表示行列，后两维分别表示距离和角度。
    glcm = skimage.feature.graycomatrix(img_gray, [1], [0], levels=16, symmetric=True, normed=True)
    glcm_feature = []
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        temp = skimage.feature.graycoprops(glcm, prop).flatten()
        glcm_feature.extend(temp)
    return glcm_feature


def compute_feature_of_batch(batchs_path):
    """
    计算一整个批次所有图片的'contrast', 'homogeneity', 'energy', 'correlation'
    """
    batchs = os.listdir(batchs_path)
    for batch in tqdm(batchs):
        batch_path = batchs_path + '/' + batch
        images = os.listdir(batch_path)
        data = pd.DataFrame()
        info = []
        for image in tqdm(images):
            img = cv2.imread(os.path.join(batch_path, image), 1)
            # 损坏的图片无法打开，直接删去
            if img is None:
                print("PIC ERROR, DELETE：" + image)
                batch_path = batchs_path + '/'
                real_file_path = batch_path + image
                os.remove(real_file_path)
                continue
            # 提取纹理特征
            texture_feature = compute_glcm(img)
            batch_id = batch
            texture_feature.append(batch_id)
            time = image[15:-4]
            texture_feature.append(time)
            info.append(texture_feature)
        data = data.append(info, ignore_index=True)
        data.columns = ['contrast', 'homogeneity', 'energy', 'correlation','batch_id', 'time']
        data_saved = save_path + '/' + batch + ".csv"
        data.to_csv(data_saved, index=True)
        print("特征储存完成...")



if __name__ == '__main__':
    batchs_path = r"F:\pythoncode\Tobacco_thesis\Data_set\Origin_Pic"
    save_path = r"F:\pythoncode\Tobacco_thesis\Data_set\Extract_Data\Texture_features"
    compute_feature_of_batch(batchs_path)