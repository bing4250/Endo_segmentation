import cv2
import numpy as np
import os
import tqdm

image_path = r'VIS/external/mask'
# ut_save = image_path.replace('out_save','max_save')
ut_save = image_path
os.makedirs(ut_save,exist_ok=True)


for idx,img in enumerate(os.listdir(image_path)):
    # 读取灰度图像
    image = cv2.imread(os.path.join(image_path,img), 0)
    # 高斯模糊
    blurred_image = cv2.GaussianBlur(image, (31, 31), 0)
    # 应用阈值操作
    _, threshold_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours_p, hierarchy_p = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个与输入图像大小相同的空白图像
    image_with_contours = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    # 增加判断，只有当有轮廓存在时才进行处理
    if len(contours_p) > 0:
        # 计算每个轮廓的面积并找到最大的轮廓
        max_area = 0
        contour_save = None
        for contour in contours_p:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                contour_save = contour
        
        # 绘制最大轮廓
        if contour_save is not None:
            cv2.drawContours(image_with_contours, [contour_save], -1, (255, 255, 255), cv2.FILLED)
        
        # 显示结果
        cv2.imwrite(os.path.join(ut_save,img),image_with_contours)
    else:
        print("No contours found.")
