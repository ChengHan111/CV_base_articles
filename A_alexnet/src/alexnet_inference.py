# -*- coding: utf-8 -*-
"""
# @file name  : alexnet_inference.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-13
# @brief      : inference demo
"""
"""
注意：
1. 模型接受4D张量
2. 弃用LRN (LRN没啥用)
3. 增加AdaptiveAvgPool2d 自适应池化 将尺寸不匹配的图都变为6x6
4. 卷积核数量有所改变
"""

import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import time
import json
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


def load_class_names(p_clsnames, p_clsnames_cn):
    """
    加载标签名
    :param p_clsnames:
    :param p_clsnames_cn:
    :return:
    """
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)
    with open(p_clsnames_cn, encoding='UTF-8') as f:  # 设置文件对象
        class_names_cn = f.readlines()
    return class_names, class_names_cn


def get_model(path_state_dict, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.alexnet() # debug模式下stepinto 两次可以看到网络结构 注意官方定义的网络结构和paper中的已经不太一样了
    pretrained_state_dict = torch.load(path_state_dict) # 加载pytorch官方预训练模型
    model.load_state_dict(pretrained_state_dict)
    model.eval() # 不在训练状态时进行eval状态

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu") # summary函数输入虚拟数据进行forward, 然后print出数据

    model.to(device)
    return model


def process_img(path_img):

    # hard code
    norm_mean = [0.485, 0.456, 0.406] # ImageNet上得到的数据,在大多数数据集中我们都使用ImageNet上统计的来的结果 R, G, B 均值和标准差
    norm_std = [0.229, 0.224, 0.225]
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)), # 中心裁剪224x224
        transforms.ToTensor(), # ToTensor会把值从[0-255]变成[0-1]区间
        transforms.Normalize(norm_mean, norm_std),
    ])

    # path --> img
    img_rgb = Image.open(path_img).convert('RGB')

    # img --> tensor
    img_tensor = img_transform(img_rgb, inference_transform)
    img_tensor.unsqueeze_(0)        # chw --> bchw
    img_tensor = img_tensor.to(device)

    return img_tensor, img_rgb


if __name__ == "__main__":

    # config
    path_state_dict = os.path.join(BASE_DIR, "..", "data", "alexnet-owt-4df8aa71.pth")
    # 两张测试图片
    # path_img = os.path.join(BASE_DIR, "..", "data", "Golden Retriever from baidu.jpg")
    path_img = os.path.join(BASE_DIR, "..", "data", "tiger cat.jpg")

    path_classnames = os.path.join(BASE_DIR, "..", "data", "imagenet1000.json")
    path_classnames_cn = os.path.join(BASE_DIR, "..", "data", "imagenet_classnames.txt")

    # load class names 加载txt和json文件
    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    # 1/5 load img
    img_tensor, img_rgb = process_img(path_img)

    # 2/5 load model
    alexnet_model = get_model(path_state_dict, True)

    # 3/5 inference  tensor --> vector
    with torch.no_grad(): # 默认情况会记录梯度，这里直接使用param的时候要防止grad
        time_tic = time.time()
        outputs = alexnet_model(img_tensor)
        time_toc = time.time()

    # 4/5 index to class names
    _, pred_int = torch.max(outputs.data, 1)
    _, top5_idx = torch.topk(outputs.data, 5, dim=1) # top-5预测, torch.topk 的返回值为 Tensor: 前k大的值, LongTensor: 前k大的值所在的位置

    pred_idx = int(pred_int.cpu().numpy())
    pred_str, pred_cn = cls_n[pred_idx], cls_n_cn[pred_idx] #找到对应的index, txt是中文文档，json是英文文档
    print("img: {} is: {}\n{}".format(os.path.basename(path_img), pred_str, pred_cn))
    print("time consuming:{:.2f}s".format(time_toc - time_tic))

    # 5/5 visualization
    plt.imshow(img_rgb)
    plt.title("predict:{}".format(pred_str))
    top5_num = top5_idx.cpu().numpy().squeeze()
    text_str = [cls_n[t] for t in top5_num]
    for idx in range(len(top5_num)):
        plt.text(5, 15+idx*30, "top {}:{}".format(idx+1, text_str[idx]), bbox=dict(fc='yellow'))
    plt.show()

