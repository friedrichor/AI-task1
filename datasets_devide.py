import os
import shutil
import random
from tqdm import tqdm
import pandas as pd


# 按类划分数据集
def datasets_div(file, content_ori, content_div):  # csv文件，原始数据集目录路径，划分后数据集目录路径
    df = pd.read_csv(file)
    df = df[['image', 'Retinopathy_grade']]
    # df = df[['Image name', 'Retinopathy grade']]
    # 按类创建处理后的图片文件夹
    classes = ['0', '1', '2', '3']
    for cls in classes:
        folder = os.path.join(content_div, cls)
        if os.path.exists(folder):
            shutil.rmtree(folder)  # 文件夹存在则删除(清空文件的目的)
        os.makedirs(folder)
    # 按类划分图片
    for index, row in df.iterrows():
        # img, grade = row[0] + '.png', row[1]
        img, grade = row[0] + '.jpg', row[1]
        shutil.copy(os.path.join(content_ori, img), os.path.join(content_div, str(grade), row[0] + '_t.jpg'))
    print('图片总量 =', len(os.listdir(content_ori)))
    for cls in classes:
        print(f'grade={cls} 图片数量 =', len(os.listdir(os.path.join(content_div, cls))))


# 按类划分数据集
def new_datasets_div(file, content_ori, content_div):  # csv文件，原始数据集目录路径，划分后数据集目录路径
    df = pd.read_csv(file)
    # 按类创建处理后的图片文件夹
    classes = ['0', '1', '2', '3', '4']
    for cls in classes:
        folder = os.path.join(content_div, cls)
        if os.path.exists(folder):
            shutil.rmtree(folder)  # 文件夹存在则删除(清空文件的目的)
        os.makedirs(folder)
    # 按类划分图片
    for index, row in tqdm(df.iterrows()):
        img, grade = row[0] + '.png', row[1]
        shutil.copy(os.path.join(content_ori, img), os.path.join(content_div, str(grade)))
    print('图片总量 =', len(os.listdir(content_ori)))
    for cls in classes:
        print(f'grade={cls} 图片数量 =', len(os.listdir(os.path.join(content_div, cls))))


def add_ori_and_new(path_ori, path_new, path_after_add, num_after_add=500):
    for cls in ['0', '1', '2', '3']:
        path_ori_cls = os.path.join(path_ori, cls)
        path_new_cls = os.path.join(path_new, cls)
        path_after_add_cls = os.path.join(path_after_add, cls)

        if os.path.exists(path_after_add_cls):
            shutil.rmtree(path_after_add_cls)  # 文件夹存在则删除(清空文件的目的)
        shutil.copytree(path_ori_cls, path_after_add_cls)

        num_add = num_after_add - len(os.listdir(path_ori_cls))
        if len(os.listdir(path_new_cls)) >= num_add:
            set_add = random.sample(os.listdir(path_new_cls), num_add)
            for img in tqdm(os.listdir(path_new_cls)):
                if img in set_add:
                    img_path = os.path.join(path_new_cls, img)
                    shutil.copy(img_path, path_after_add_cls)
        else:
            for img in tqdm(os.listdir(path_new_cls)):
                img_path = os.path.join(path_new_cls, img)
                shutil.copy(img_path, path_after_add_cls)


# 划分数据集，分为两类：(1)训练集+验证集;(2)测试集
def split_train_test(content_div, content_split, rate_test=0.1):
    for cls in os.listdir(content_div):
        content_cls = os.path.join(content_div, cls)  # 当前类别的目录路径
        num_cls = len(os.listdir(content_cls))  # 当前类别中图片数量
        test_set = random.sample(os.listdir(content_cls), int(num_cls*rate_test))  # 随机取 num_cls*rate_test 个图片作为测试集

        content_train_cls = os.path.join(content_split, 'train', cls)
        content_test_cls = os.path.join(content_split, 'test', cls)
        if os.path.exists(content_train_cls):  # 清空
            shutil.rmtree(content_train_cls)
        os.makedirs(content_train_cls)
        if os.path.exists(content_test_cls):  # 清空
            shutil.rmtree(content_test_cls)
        os.makedirs(content_test_cls)

        print(f'正在划分grade={cls}的数据集...')
        for img in tqdm(os.listdir(content_cls)):
            if img in test_set:
                shutil.copy(os.path.join(content_cls, img), content_test_cls)
            else:
                shutil.copy(os.path.join(content_cls, img), content_train_cls)


# if __name__ == '__main__':
    # file = 'data_ori/Mess1_annotation_train.csv'
    # content_ori = 'data_ori/datasets/'  # 原始图片数据集
    # content_div = 'data_ori/datasets_div/'  # 分类后数据集
    # content_split = 'data_ori/'
    # if not os.path.exists(content_div):
    #     os.mkdir(content_div)
    # datasets_div(file, content_ori, content_div)
    # split_train_test(content_div, content_split)

    ### divide new_extra_datasets
    # file = 'new_extra_datasets/train.csv'
    # content_ori = 'new_extra_datasets/train_images'
    # content_div = 'new_extra_datasets/new_extra_div'
    # new_datasets_div(file, content_ori, content_div)

    ### add new_extra_datasets and data_ori/datasets_div
    # data_ori = 'data_ori/datasets_div'
    # data_new = 'new_extra_datasets/new_extra_div'
    # path_after_add = 'new_extra_datasets/after_add_ori_and_extra'
    # add_ori_and_new(data_ori, data_new, path_after_add)

    # content_div = 'new_extra_datasets/add_ori_and_extra_after_pro'
    # content_split = 'new_extra_datasets/add_ori_and_extra_pro_split'
    # split_train_test(content_div, content_split, rate_test=0.1)

    # file = r'E:\pythonProjects\course_AIpractice\task1\B. Disease Grading\2. Groundtruths\b. IDRiD_Disease Grading_Testing Labels.csv'
    # content_ori = r'E:\pythonProjects\course_AIpractice\task1\B. Disease Grading\1. Original Images\b. Testing Set'
    # content_div = r'E:\pythonProjects\course_AIpractice\task1\B. Disease Grading\test_div'
    # datasets_div(file, content_ori, content_div)