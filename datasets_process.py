import os
import shutil
import random
from tqdm import tqdm
import cv2
import numpy as np

import params


def crop_image_from_gray(img, boundary=9):
    """
    :param img: cv2读取的图片，BGR格式
    :param boundary: mask，颜色大于boundary的去除，去除过多的冗余黑色背景
    :return: 处理后的图片
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray_img > boundary
    check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if check_shape == 0:  # image is too dark so that we crop out everything,
        print('check_shape == 0')
        return img  # return original image
    else:
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = cv2.merge((img1, img2, img3))
    return img


def circle_crop(img, sigmaX=50):
    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (512, 512))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 50), -4, 128)
    return img


def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (512, 512))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image


def images_process(data_ori, data_pro, img_size=512):
    """
    :param data_ori: 原数据集目录
    :param data_pro: 新数据集目录
    :param img_size: resize后的图片size
    :return: None
    """
    for cls in os.listdir(data_ori):
        data_ori_cls = os.path.join(data_ori, cls)
        data_pro_cls = os.path.join(data_pro, cls)

        if os.path.exists(data_pro_cls):
            shutil.rmtree(data_pro_cls)
        os.makedirs(data_pro_cls)

        for img_name in tqdm(os.listdir(data_ori_cls)):
            img_path = os.path.join(data_ori_cls, img_name)
            img = cv2.imread(img_path)
            # Ben
            img_pro = load_ben_color(img, int(img_size/10))
            # Jung
            # img_pro = circle_crop(img, int(img_size/10))

            cv2.imwrite(os.path.join(data_pro_cls, img_name.split('.')[0] + '.png'), img_pro)


if __name__ == '__main__':
    # 训练集
    data_ori = params.path_train
    data_pro = params.path_train_pro
    images_process(data_ori, data_pro)
