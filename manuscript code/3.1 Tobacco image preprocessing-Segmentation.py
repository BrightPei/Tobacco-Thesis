"""
对图片中的烟叶进行掩膜,使用GrabCut算法
"""

import numpy as np
import cv2
import os
from tqdm import tqdm
import time

class PicCut(object):
    def __init__(self,ori_file_path,processed_file_path):
        """
        :param self.ori_pic_path: 待处理图片储存的总路径
        :param self.processed_file_path: 已处理图片储存的总路径
        """
        self.ori_file_path = ori_file_path
        self.processed_file_path = processed_file_path

    #进行图片掩膜
    def pic_spilt(self,img):
        mask = np.zeros(img.shape[:2], np.uint8)
        # 创建以0填充的前景和背景模型：
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        # 在实现GrabCut算法前，先用一个标识出想要隔离的对象的矩形来初始化它，这个矩形我们用下面的一行代码定义（x,y,w,h）分别为左上角坐标和宽度，高度：
        rect = (1, 1, 1920, 1080)
        # 接下来用指定的空模型和掩摸来运行GrabCut算法
        mask, bgdModel, fgdModel = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5,cv2.GC_INIT_WITH_RECT)  # 5是指算法的迭代次数。
        # 然后，我们再设定一个掩模，用来过滤之前掩模中的值（0-3）。值为0和2的将转为0，值为1和3的将转化为1，这样就可以过滤出所有的0值像素（背景）。
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        img_spilt = img * mask2[:, :, np.newaxis]
        return img_spilt

    #按照批次对图像进行处理
    def pic_cut_for_batch(self):
        print("图像处理中...")
        time.sleep(1)
        batch_paths = os.listdir(self.ori_file_path)
        for batch_path in tqdm(batch_paths):
            time.sleep(1)
            print()
            print(f"{batch_path}批次图像处理中...")
            pic_saved = os.path.join(self.processed_file_path,batch_path)
            if not os.path.exists(pic_saved):
                os.makedirs(pic_saved)
            batch_path = os.path.join(self.ori_file_path,batch_path)
            pic_paths = os.listdir(batch_path)
            for pic_path in tqdm(pic_paths):
                img = cv2.imread(batch_path + '/' +pic_path)
                # 有些损坏的图片无法打开，直接删去
                if img is None:
                    print("PIC ERROR, DELETE：" + pic_path)
                    real_file_path = batch_path + '/' + pic_path
                    os.remove(real_file_path)
                    return
                img_spilt = self.pic_spilt(img)
                img_spilt_saved_path = os.path.join(pic_saved,pic_path)
                cv2.imwrite(img_spilt_saved_path, img_spilt)
            print(f"{batch_path}路径下的图像全部处理完毕")
        print(f"图像全部处理完成")

if __name__ == '__main__':
    # 图片读取路径，该文件夹下有若干批次
    ori_file_path = r'F:\pythoncode\Tobacco_thesis\Data_set\Origin_Pic'
    # 图片存放位置，该文件夹下会自动生成若干个存有图片的文件
    processed_file_path = r"F:/pythoncode/Tobacco_thesis/Data_set/Segmentation_Pic"
    PC = PicCut(ori_file_path, processed_file_path)
    PC.pic_cut_for_batch()