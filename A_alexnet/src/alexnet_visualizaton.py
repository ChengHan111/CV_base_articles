# -*- coding: utf-8 -*-
"""
# @file name  : alexnet_visualization.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-14
# @brief      : alexnet_visualization
# 卷积核以及feature map可视化代码
为什么paper中只展示第一层可视化的代码是因为往后面的kernel_size很小，而且逐渐抽象，因此再展示就没有意义了
"""
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torchvision.models as models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    log_dir = os.path.join(BASE_DIR, "..", "results")
    # ----------------------------------- kernel visualization, kernel的可视化-----------------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_kernel") # 定义名字和位置

    # 两种方法
    # m1
    # alexnet = models.alexnet(pretrained=True)

    # m2 预训练模型参数加载
    path_state_dict = os.path.join(BASE_DIR, "..", "data", "alexnet-owt-4df8aa71.pth")
    alexnet = models.alexnet()
    pretrained_state_dict = torch.load(path_state_dict)
    alexnet.load_state_dict(pretrained_state_dict)

    kernel_num = -1
    vis_max = 1 # 可以设置为0-4，可视化卷积层的数量，因为AlexNet里只有五个卷积层因此只能到4 设置为1时显示前两层卷积层
    # 只寻找对应的conv层，非conv层不进行判断
    for sub_module in alexnet.modules():
        if not isinstance(sub_module, nn.Conv2d):
            continue
        kernel_num += 1
        if kernel_num > vis_max:
            break

        kernels = sub_module.weight
        c_out, c_int, k_h, k_w = tuple(kernels.shape)

        # 拆分channel为R,G,B分开
        for o_idx in range(c_out):
            kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)  # 获得(3, h, w), 但是make_grid需要 BCHW，这里拓展C维度变为（3， 1， h, w）
            kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_int)
            writer.add_image('{}_Convlayer_split_in_channel'.format(kernel_num), kernel_grid, global_step=o_idx)

        kernel_all = kernels.view(-1, 3, k_h, k_w)  # 3, h, w
        kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)  # c, h, w
        writer.add_image('{}_all'.format(kernel_num), kernel_grid, global_step=620)

        print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))

    # ----------------------------------- feature map visualization feature map的可视化-----------------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_feature map")

    # 数据
    path_img = os.path.join(BASE_DIR, "..", "data", "tiger cat.jpg")  # your path to image
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    norm_transform = transforms.Normalize(normMean, normStd)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform
    ])

    img_pil = Image.open(path_img).convert('RGB')
    img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)  # chw --> bchw

    # 模型
    # alexnet = models.alexnet(pretrained=True)

    # forward
    # features在init中定义了一个sequential,其第0个就是第一个卷积层
    convlayer1 = alexnet.features[0]
    fmap_1 = convlayer1(img_tensor)

    # 预处理 feature map的shape和想要的维度不一样，要对前两个维度进行交换，这是在make_grid中需要注意的点
    fmap_1.transpose_(0, 1)  # bchw=(1, 64, 55, 55) --> (64, 1, 55, 55)
    # 制作网格图像
    fmap_1_grid = vutils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)

    writer.add_image('feature map in conv1', fmap_1_grid, global_step=620)
    writer.close()
