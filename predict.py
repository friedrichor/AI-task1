import os
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import params
from models import *


def main(args):
    print(args)
    device = params.device
    print(f"using {device} device.")

    num_classes = params.num_classes
    img_size = params.img_size
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
        # [transforms.Resize(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = args.path_json
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = convnext(num_classes).to(device)
    # load model weights
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    test_path = args.path_test
    num_acc = 0
    num_all = 0
    label_true = []
    label_predict = []
    for cls in os.listdir(test_path):
        num_all += len(os.listdir(os.path.join(test_path, cls)))
        for img_path in os.listdir(os.path.join(test_path, cls)):
            label_true.append(int(cls))
            img_path = os.path.join(test_path, cls, img_path)
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            plt.imshow(img)
            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cls = torch.argmax(predict).numpy()
                label_predict.append(int(predict_cls))

                # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cls)],
                #                                              predict[predict_cls].numpy())
                # print(print_res)
                # for i in range(len(predict)):
                #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))

                if str(predict_cls) == cls:
                    num_acc += 1

    print('num of test datasets =', num_all)
    print('acc =', num_acc / num_all)
    category_show(label_true, label_predict)


# ???????????????????????????????????????f1-score???????????????????????????
def category_show(y_true, y_predict):
    target_names = ['0', '1', '2', '3']
    print(classification_report(y_true, y_predict, target_names=target_names))
    cm = confusion_matrix(y_true, y_predict)
    cm_display = ConfusionMatrixDisplay(cm).plot()


def parse_opt():
    parser = argparse.ArgumentParser()
    # ????????????
    parser.add_argument('--weights', nargs='+', type=str, default=params.weights, help='model weights path')
    # ???????????????
    parser.add_argument('--path_test', type=str, default=params.path_test, help='test datasets path')
    parser.add_argument('--path_json', type=str, default=params.path_json, help='class_indice.json path')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
