import torch
import torch.nn as nn
from functions import *

class ComplexSequential(nn.Sequential):
    def forward(self, input_r, input_i):
        for module in self._modules.values():
            input_r, input_i = module(input_r, input_i)
        return input_r, input_i
    

class ComplexDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input_r, input_i):
        return c_dropout(input_r, input_i, self.p, self.training, self.inplace)

class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size=(1, 1)):
        super(ComplexAdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, input_r, input_i):
        return c_adapavgool(input_r, input_i, self.output_size)
    
class ComplexDropout2d(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout2d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input_r, input_i):
        return c_dropout2d(input_r, input_i, self.p, self.training, self.inplace)
    
class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input_r, input_i):
        return c_maxpool(input_r, input_i, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
    
class ComplexReLU(nn.Module):
    def forward(self, input_r, input_i):
        return c_relu(input_r, input_i)
    
class ComplexTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation):
        super(ComplexTranspose2d, self).__init__()

        self.conv_tran_r = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        self.conv_tran_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)

    def forward(self, input_r, input_i):
        return self.conv_tran_r(input_r) - self.conv_tran_i(input_i), \
            self.conv_tran_r(input_i) + self.conv_tran_i(input_r)
    
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()

        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_r, input_i):
        return self.conv_r(input_r) - self.conv_i(input_i), \
            self.conv_r(input_i) + self.conv_i(input_r)
    
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super(ComplexLinear, self).__init__()

        self.linear_r = nn.Linear(in_features, out_features, bias)
        self.linear_i = nn.Linear(in_features, out_features, bias)

    def forward(self, input_r, input_i):
        return self.linear_r(input_r) - self.linear_i(input_i), \
            self.linear_r(input_i) + self.linear_i(input_r)
    
class NaiveComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input_r, input_i):
        return self.bn_r(input_r), self.bn_i(input_i)
    
class NaiveComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(NaiveComplexBatchNorm2d, self).__init__()
        self.bn_r = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input_r, input_i):
        return self.bn_r(input_r), self.bn_i(input_i)
    
class _ComplexBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            self.running_covar[:, 0] = 1.4142
            self.running_covar[:, 1] = 1.4142
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar[:, 0] = 1.4142
            self.running_covar[:, 1] = 1.4142
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

class ComplexBatchNorm2d(_ComplexBatchNorm):
    def forward(self, input_r, input_i):
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum


        if self.training:
            mean_r = input_r.mean([0, 2, 3])
            mean_i = input_i.mean([0, 2, 3])
            mean = torch.stack([mean_r, mean_i], dim=1)

            with torch.no_grad():
                self.running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * mean    
            
            input_r = input_r - mean_r[None, :, None, None]
            input_i = input_i - mean_i[None, :, None, None]

            n  = input_r.numel() / input_r.size(1)
            crr = 1./n*input_r.pow(2).sum(dim=[0,2,3])+self.eps
            cii = 1./n*input_i.pow(2).sum(dim=[0,2,3])+self.eps
            cri = 1./n*input_r.mul(input_i).sum(dim=[0,2,3])

            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]
        else:
            mean = self.running_mean
            crr = self.running_covar[:,0] + self.eps  
            cii = self.running_covar[:,1] + self.eps
            cri = self.running_covar[:,2]

            input_r = input_r - mean[:,0][None, :, 0, None, None]
            input_i = input_i - mean[:,1][None, :, 1, None, None]
        

        det = crr*cii-cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(cii+crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (cii + s) * inverse_st
        Rii = (crr + s) * inverse_st
        Rri = -cri * inverse_st

        input_r, input_i = Rrr[None,:,None,None]*input_r+Rri[None,:,None,None]*input_i, \
                           Rii[None,:,None,None]*input_i+Rri[None,:,None,None]*input_r

        if self.affine:
            input_r, input_i = self.weight[None,:,0,None,None]*input_r+self.weight[None,:,2,None,None]*input_i+\
                               self.bias[None,:,0,None,None], \
                               self.weight[None,:,2,None,None]*input_r+self.weight[None,:,1,None,None]*input_i+\
                               self.bias[None,:,1,None,None]

        return input_r, input_i


class ComplexBatchNorm1d(_ComplexBatchNorm):

    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        assert(len(input_r.shape) == 2)
        #self._check_input_dim(input)

        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None: 
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  
                    exponential_average_factor = self.momentum

        if self.training:

            mean_r = input_r.mean(dim=0)
            mean_i = input_i.mean(dim=0)
            mean = torch.stack((mean_r,mean_i),dim=1)

            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

            input_r = input_r-mean_r[None, :]
            input_i = input_i-mean_i[None, :]


            n = input_r.numel() / input_r.size(1)
            crr = input_r.var(dim=0,unbiased=False)+self.eps
            cii = input_i.var(dim=0,unbiased=False)+self.eps
            cri = (input_r.mul(input_i)).mean(dim=0)

            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]

        else:
            mean = self.running_mean
            crr = self.running_covar[:,0]+self.eps
            cii = self.running_covar[:,1]+self.eps
            cri = self.running_covar[:,2]
            input_r = input_r-mean[None,:,0]
            input_i = input_i-mean[None,:,1]

        det = crr*cii-cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(cii+crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (cii + s) * inverse_st
        Rii = (crr + s) * inverse_st
        Rri = -cri * inverse_st

        input_r, input_i = Rrr[None,:]*input_r+Rri[None,:]*input_i, \
                           Rii[None,:]*input_i+Rri[None,:]*input_r

        if self.affine:
            input_r, input_i = self.weight[None,:,0]*input_r+self.weight[None,:,2]*input_i+\
                               self.bias[None,:,0], \
                               self.weight[None,:,2]*input_r+self.weight[None,:,1]*input_i+\
                               self.bias[None,:,1]

        del crr, cri, cii, Rrr, Rii, Rri, det, s, t
        return input_r, input_i


