import os
import sys
from pathlib import Path
import torch

from utils import FocalLoss


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 4
img_size = 384

# 损失函数
loss_function = FocalLoss(num_classes)

# 模型保存文件夹
path_weights = '/content/drive/MyDrive/weights_convnext_multi'

# 预测所需模型权重
weights = 'weights/convnext-data4-pre384-re384-ac83-cls91-31-80-87.pth'

# 数据集&分类标签 路径
path_train = '/content/drive/MyDrive/train4'
path_train_pro = ROOT / 'train_pro'
path_test = '../../test'
path_test_pro = ROOT / 'test_pro'
path_json = ROOT / 'class_indices.json'