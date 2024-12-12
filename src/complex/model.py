import torch
import torch.nn as nn
from layers import *
from functions import *
from shift import *
from complexUtils import *


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride = 1, downsample = None, 
                 groups = 1, base_width = 64, dilation = 1, norm_layer = None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = ComplexBatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = ComplexConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = ComplexReLU()
        self.conv2 = ComplexConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x_r, x_i):
        identity_r, identity_i = x_r, x_i

        out_r, out_i = self.conv1(x_r, x_i)
        out_r, out_i = self.bn1(out_r, out_i)
        out_r, out_i = self.relu(out_r, out_i)

        out_r, out_i = self.conv2(out_r, out_i)
        out_r, out_i = self.bn2(out_r, out_i)

        if self.downsample is not None:
            identity_r, identity_i = self.downsample(x_r, x_i)

        out_r += identity_r
        out_i += identity_i
        out_r, out_i = self.relu(out_r, out_i)

        return out_r, out_i
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride = 1, downsample = None, 
                 groups = 1, base_width = 64, dilation = 1, norm_layer = None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = ComplexBatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = ComplexConv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = ComplexConv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = ComplexConv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = ComplexReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x_r, x_i):
        identity_r, identity_i = x_r, x_i

        out_r, out_i = self.conv1(x_r, x_i)
        out_r, out_i = self.bn1(out_r, out_i)
        out_r, out_i = self.relu(out_r, out_i)

        out_r, out_i = self.conv2(out_r, out_i)
        out_r, out_i = self.bn2(out_r, out_i)
        out_r, out_i = self.relu(out_r, out_i)

        out_r, out_i = self.conv3(out_r, out_i)
        out_r, out_i = self.bn3(out_r, out_i)

        if self.downsample is not None:
            identity_r, identity_i = self.downsample(x_r, x_i)

        out_r += identity_r
        out_i += identity_i
        out_r, out_i = self.relu(out_r, out_i)

        return out_r, out_i
    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 2, zero_init_residual = False,
                 groups = 1, width_per_group = 64, replace_stride_with_dilation = None,
                 norm_layer = None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = ComplexBatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = ComplexConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = ComplexReLU()
        self.maxpool = ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = ComplexAdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ComplexConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, padding=0),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return Sequential_complex(*layers)
    
    def forward(self, x_r, x_i):
        x_r, x_i = self.conv1(x_r, x_i)
        x_r, x_i = self.bn1(x_r, x_i)
        x_r, x_i = self.relu(x_r, x_i)
        x_r, x_i = self.maxpool(x_r, x_i)

        x_r, x_i = self.layer1(x_r, x_i)
        x_r, x_i = self.layer2(x_r, x_i)
        x_r, x_i = self.layer3(x_r, x_i)
        x_r, x_i = self.layer4(x_r, x_i)

        x_r, x_i = self.avgpool(x_r, x_i)
        x_r, x_i = torch.flatten(x_r, 1), torch.flatten(x_i, 1)
        x_r, x_i = self.fc(x_r, x_i)

        return x_r, x_i
    
def _complexresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def complexresnet50(pretrained=False, progress=True, **kwargs):
    return _complexresnet('complexresnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def complexresnet18(pretrained=False, progress=True, **kwargs):
    return _complexresnet('complexresnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def complexresnet34(pretrained=False, progress=True, **kwargs):
    return _complexresnet('complexresnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

if __name__ == '__main__':
    model = complexresnet50()
    print(model)