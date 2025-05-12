import os
import numpy as np
import cv2
import shutil
from tqdm import tqdm
import pandas as pd
import datetime
import torch
from PIL import Image
# from seg_eva import __surface_distances, recall, precision
# from hce_metric_main import compute_hce

def dict_to_excel(metric_dict, excel_path):
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    df.to_excel(excel_path)

# dice系数
def dc(result, reference):
    r"""
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    .. math::

        DC=\frac{2|A\cap B|}{|A|+|B|}

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc

# # 豪斯多夫距离
# def hd(result, reference, voxelspacing=None, connectivity=1):
#     try:
#         hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
#         hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
#     except:
#         hd = 0
#         return hd

#     hd = max(hd1, hd2)
#     return hd

# 杰卡德相似系数
def jc(result, reference):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)

    jc = float(intersection) / float(union)

    return jc

# 平均表面距离
# def asd(result, reference, voxelspacing=None, connectivity=1):
#     try:
#         sds = __surface_distances(result, reference, voxelspacing, connectivity)
#     except:
#         asd = 0
#         return asd
#     asd = sds.mean()
#     return asd

# 平均对称表面距离
# def assd(result, reference, voxelspacing=None, connectivity=1):
#     assd = np.mean(
#         (asd(result, reference, voxelspacing, connectivity), asd(reference, result, voxelspacing, connectivity)))
#     return assd

# 相对体积差
def RVD(result, reference):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    vol1 = np.count_nonzero(result)
    vol2 = np.count_nonzero(reference)

    if 0 == vol2:
        raise RuntimeError('The second supplied array does not contain any binary object.')

    return 100 * np.abs(vol1 / vol2 - 1)

# def F1_score(result,reference):
#     pre = precision(result, reference)
#     sen = recall(result, reference)
#     f1_score = (1+0.3)*pre*sen/(0.3*pre+sen + 1e-4)

#     return f1_score

def MAE(result,reference):
    mae_sum = np.sum(np.abs(result - reference)) * 1.0 / ((reference.shape[0] * reference.shape[1] * 255.0) + 1e-4)

    return mae_sum

def conformity(result, reference):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    tp = np.count_nonzero(result & reference)

    fp = np.count_nonzero(result ^ reference)
    try:
        con = (1-float(fp)/tp)
    except ZeroDivisionError:
        con = 0.0

    return con

if __name__ == '__main__':
    # 存放模型预测结果的路径
    pre_root = r'VIS/out_save/Unet'
    # 包含测试图像路径的文本文件路径
    test_source = r'Data/test/images'
    # 保存评估结果和可视化结果的根目录路径
    save_root = r'VIS/badexam_visual/'+os.path.split(pre_root)[-1]
    # save_root = os.path.join(save_root,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_root, exist_ok=True)
    dice_ut = []
    jc_ut = []
    img_lst = [os.path.join(test_source,x) for x in os.listdir(test_source)]
    for idx,img_path in enumerate(tqdm(img_lst)):
        # 根据图像路径生成对应的掩码图像路径，通过将 image替换为mask。
        mask_path = img_path.replace('images','masks')

        # 读取当前图像的彩色图像
        # img = cv2.imread(img_path)
        # img_ut = img.copy()
        img = Image.open(img_path)
        # img_ut = img.copy()
        img_ut = img.crop((0,img.size[1]/10,img.size[0]-img.size[0]/15,img.size[1]-img.size[1]/10))
        img_ut = np.array(img_ut)
        # 读取当前图像对应的掩码图像，灰度模式
        mask = Image.open(mask_path)
        mask = mask.convert('L')
        mask = mask.crop((0,mask.size[1]/10,mask.size[0]-mask.size[0]/15,mask.size[1]-mask.size[1]/10))
        mask = np.array(mask)
        mask1 = np.array(mask!=0,dtype=np.uint8)

        mask1 = mask1 * 255

        kernel = np.ones((9,9),np.uint8)

        # # 膨胀
        # mask1 = cv2.dilate(mask1,kernel,iterations=1)
        # mask2 = cv2.dilate(mask2,kernel,iterations=1)

        # # 腐蚀
        # mask1 = cv2.erode(mask1,kernel,iterations=1)
        # mask2 = cv2.erode(mask2,kernel,iterations=1)
        
        # 高斯模糊
        mask1 = cv2.GaussianBlur(mask1, (31, 31), 0)

        # 应用阈值操作
        _, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
       
        # 获取当前图像的高度和宽度
        h_img ,w_img = img.size[1],img.size[0]

        #再画子宫的
        infer_path = os.path.join(pre_root,img_path.split('/')[-1])

        infer = cv2.imread(infer_path,0)

        # 读取当前图像的模型预测结果，以灰度模式读取
        if infer is not None:
            # 膨胀
            infer = cv2.dilate(infer,kernel,iterations=1)

            # 腐蚀
            infer = cv2.erode(infer,kernel,iterations=1)

            # 高斯模糊
            infer = cv2.GaussianBlur(infer, (15, 15), 0)

            # 应用阈值操作
            _, infer = cv2.threshold(infer, 127, 255, cv2.THRESH_BINARY)

            dice = dc(infer, mask1)
            dice_ut.append(dice)
            jc_coffidence = jc(infer,mask1)
            jc_ut.append(jc_coffidence)

            h,w = img.size[1],img.size[0]
            # img_ori = img.copy()
            if dice < 0.95:
                contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  #    Find Contour
                if len(contours) > 0:  # 增加判断，只有当有轮廓存在时才填充轮廓！
                    cv2.drawContours(img_ut, contours, -1, (0, 0, 255), 2)

                contours_p, hierarchy_p = cv2.findContours(infer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #    Find Contour
                if len(contours_p) > 0:  # 增加判断，只有当有轮廓存在时才填充轮廓！
                    cv2.drawContours(img_ut, contours_p, -1, (0, 255, 0), 2)


                cv2.putText(img_ut, f"dice={dice:.3f}", (0, h-int(h/10)-100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
                cv2.imwrite(os.path.join(save_root, infer_path.split('/')[-1]),img_ut)

    dice_ut_avg = sum(dice_ut)/len(img_lst)
    jc_ut_avg = sum(jc_ut)/len(img_lst)
    txt_path = save_root + '/dice.txt' 
    with open(txt_path,'a') as f:
        txt = f'dice_ut : {dice_ut_avg}\n'
        txt += f'jc_ut : {jc_ut_avg}\n'

        f.write(txt+'\n')



