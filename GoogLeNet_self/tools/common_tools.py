# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-02-14
# @brief      : 通用函数
"""
import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.models as models


def get_vgg16(device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    # model = models.vgg16()
    # pretrained_state_dict = torch.load(path_state_dict)
    # model.load_state_dict(pretrained_state_dict)
    # remember to add aux_logits=True. Or there will be no additional aux_logits in the model.
    model = torch.hub.load('pytorch/vision:v0.9.0', 'googlenet', pretrained=True, aux_logits=True)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model
