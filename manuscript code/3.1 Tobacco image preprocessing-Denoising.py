"""
使用双边滤波
"""
import cv2
import matplotlib.pyplot as plt
img1 = cv2.imread(r"F:\Tobacco_thesis\Image_processing\Tobacco.jpg")

img2 = cv2.bilateralFilter(src=img1, d=9, sigmaColor=75, sigmaSpace=75)
plt.xticks([])
plt.yticks([])
plt.imshow(img2[:,:,::-1])
plt.show()

cv2.imwrite(r"F:\pythoncode\Tobacco_thesis\Image_processing\Image_blur\Tobacco.jpg", img2)

