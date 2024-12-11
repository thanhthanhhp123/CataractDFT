from pickle import TRUE
import torch
import torch.nn.functional as F
import torch.nn as nn


def c_relu(input_r, input_i):
    return F.relu(input_r), F.relu(input_i)


def c_maxpool(input_r, input_i, kernel_size = 2, stride = 2, padding = 0,
             dilation = 1, ceil_mode = False, return_indices = False):
    return F.max_pool2d(input_r, kernel_size, stride, padding, dilation, ceil_mode, return_indices), \
        F.max_pool2d(input_i, kernel_size, stride, padding, dilation, ceil_mode, return_indices)


def c_adapavgool(input_r, input_i, output_size = [1, 1]):
    return F.adaptive_avg_pool2d(input_r, output_size), \
        F.adaptive_avg_pool2d(input_i, output_size)

def c_cat(combine1_r, combine1_i, combine2_r, combine2_i, dim=1):
    return torch.cat((combine1_r, combine2_r), dim), torch.cat((combine1_i, combine2_i), dim)

def c_up(img_r, img_i, scale_factor=2):
    up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
    return up(img_r), up(img_i)

def c_dropout(input_r, input_i, p=0.5, training=True, inplace=False):
    return F.dropout(input_r, p, training, inplace), F.dropout(input_i, p, training, inplace)

def c_dropout2d(input_r, input_i, p=0.5, training=True, inplace=False):
    return F.dropout2d(input_r, p, training, inplace), F.dropout2d(input_i, p, training, inplace)


