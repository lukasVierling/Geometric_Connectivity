'''
This file was adapted from the original MMC github repository.
Authors: Ekdeep Singh Lubana et al.
Date: 09.04.2026
GitHub: https://github.com/EkdeepSLubana/MMC
Paper: https://arxiv.org/pdf/2211.08422
License: MIT License
Modifcation: Support Symmetric Kernels for ResNet18
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch

#We calculate the symmetric kernels according to the formula in the paper

#For definitions of Symmetries please refer to Appendix A of the paper
class WeightSymmetry(nn.Module):
    # vanilla weight doesn't change anything
    def forward(self, weight):
        return weight

class HorizontalFlipSymmetry(WeightSymmetry):
    def forward(self, weight):
        # use the formula for {id, H}
        return 0.5*(weight + torch.flip(weight, dims=[3]))

class VerticalFlipSymmetry(WeightSymmetry):
    def forward(self, weight):
        #use the formula for {id, V}
        return 0.5*(weight + torch.flip(weight, dims=[2]))

class HVFlipSymmetry(WeightSymmetry):
    def forward(self, weight):
        #use the formula for {id, H, V, HV}
        return 0.25*(weight + torch.flip(weight, dims=[2]) + torch.flip(weight, dims=[3]) + torch.flip(weight, dims=[2, 3]))

class Rot90Symmetry(WeightSymmetry):
    def forward(self, weight):
        #use the formula for {id, R, R^2, R^3}
        return 0.25*(weight + torch.rot90(weight, k=1, dims=[2, 3]) + torch.rot90(weight, k=2, dims=[2, 3]) + torch.rot90(weight, k=3, dims=[2, 3]))

#mapping from symmetry abbreviation to kernel implementation
SYMMETRY_CLASSES = {
    'vanilla': WeightSymmetry,
    'hflip': HorizontalFlipSymmetry,
    'vflip': VerticalFlipSymmetry,
    'hvflip': HVFlipSymmetry,
    'rot90': Rot90Symmetry
}

class SymmetricConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=False, symmetry='vanilla', **kwargs):
        super(SymmetricConv2d, self).__init__()
        #create a convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, **kwargs)
        #create respective kernel symmetry
        self.symmetry = SYMMETRY_CLASSES[symmetry]()
        
    def forward(self, x):
        #get the effective weights (K in paper)
        weight = self.symmetry(self.conv.weight)
        # do the convolution with effective (symmetric) weights instead of vanilla weights
        return F.conv2d(x, weight, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
    
def replace_sym_conv_with_normal(module):
    for name, child in list(module._modules.items()):
        if isinstance(child, SymmetricConv2d):
            #get the SymmetricConv2d layer
            conv_orig = child.conv
            #create new convolution layer
            new_conv = nn.Conv2d(
                conv_orig.in_channels,
                conv_orig.out_channels,
                conv_orig.kernel_size,
                stride=conv_orig.stride,
                padding=conv_orig.padding,
                dilation=conv_orig.dilation,
                groups=conv_orig.groups,
                bias=(conv_orig.bias is not None)
            )
            # get the symmetric Kernel K
            effective_weight = child.symmetry(conv_orig.weight).detach().clone()
            # replace K' with K
            new_conv.weight.data.copy_(effective_weight)
            if conv_orig.bias is not None:
                new_conv.bias.data.copy_(conv_orig.bias.data)
            module._modules[name] = new_conv
        else:
            replace_sym_conv_with_normal(child)

def eval_symmetry(weight, symmetry):

    weight_normalized = weight / torch.norm(weight, p="fro")
    #get all symmetry transformations according to formula
    if symmetry == "h":
        transforms = [lambda x: torch.flip(x, [-1])]
    elif symmetry == "v":
        transforms = [lambda x: torch.flip(x, [-2])]
    elif symmetry == "hv":
        transforms = [lambda x: torch.flip(x, [-1]), lambda x: torch.flip(x, [-2])]
    elif symmetry == "rot90":
        transforms= [lambda x: torch.rot90(x, k=1, dims=(-2,-1)), lambda x: torch.rot90(x, k=2,dims=(-2,-1)), lambda x: torch.rot90(x, k=3,dims=(-2,-1))]
    elif symmetry == "total":
        transforms = [
            lambda x: torch.rot90(x, k=1, dims=[2, 3]),
            lambda x: torch.rot90(x, k=2, dims=[2, 3]),
            lambda x: torch.rot90(x, k=3, dims=[2, 3]),
            lambda x: torch.flip(x, [2]),
            lambda x: torch.flip(x, [3]),
            lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), [2]),
            lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), [3])
        ]
    delta = 0
    # sum_{T in D}
    for transform in transforms:
        # T(K)
        weight_transformed = transform(weight_normalized)
        # T(K) - K
        diff = weight_transformed - weight_normalized
        # take fro norm
        delta += torch.norm(diff, p="fro")
    # / 2*|D|
    avg_delta = delta/(2*len(transforms))
    # 1- ...
    S_K = 1-avg_delta
    return S_K.item()


def accumulate_symmetry(module):
    horizontal_symmetry = 0
    vertical_symmetry = 0
    hv_symmetry = 0
    rot90_symmetry = 0
    total_symmetry = 0
    mean_weight_total_symmetry = 0
    counter = 0
    for name, child in list(module._modules.items()):
        if isinstance(child, SymmetricConv2d):
            conv_orig = child.conv
            #get the actual kernel (usually the same because we only use the function for vanilla CNNs)
            weight = child.symmetry(conv_orig.weight).detach().clone()
            #get the kxk mean kernel
            mean_weight = weight.mean(dim=(0,1), keepdim=True)
            #get all symmetric metrics
            mean_weight_total_symmetry += eval_symmetry(mean_weight, "total")
            horizontal_symmetry += eval_symmetry(weight, "h")
            vertical_symmetry += eval_symmetry(weight, "v")
            hv_symmetry += eval_symmetry(weight, "hv")
            rot90_symmetry += eval_symmetry(weight, "rot90")
            total_symmetry += eval_symmetry(weight, "total")
            counter += 1
        else:
            h, v, hv, rot90, tot, mean_tot, cntr = accumulate_symmetry(child)
            horizontal_symmetry += h
            vertical_symmetry += v
            hv_symmetry += hv
            rot90_symmetry += rot90
            total_symmetry += tot
            mean_weight_total_symmetry += mean_tot
            counter += cntr

    return horizontal_symmetry, vertical_symmetry, hv_symmetry, rot90_symmetry, total_symmetry, mean_weight_total_symmetry, counter

def evaluate_symmetry(module):
    h,v,hv,rot90, tot, mean_tot, counter = accumulate_symmetry(module)
    
    print(f"counted {counter} convolutions")

    return h/counter, v/counter, hv/counter, rot90/counter, tot/counter, mean_tot/counter

### In the following code we exchanged the Conv2d layers with our new SymmetricConv2d layers to allow for ResNet architectures that support symmetric convolutions:
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, symmetry='vanilla'):
        super(BasicBlock, self).__init__()
        self.use_shortcut = stride != 1 or in_planes != self.expansion*planes
        self.conv1 = SymmetricConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, symmetry=symmetry)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = SymmetricConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, symmetry=symmetry)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        
        self.shortcut_conv = nn.Sequential()
        if self.use_shortcut:
            self.shortcut_conv = SymmetricConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, symmetry=symmetry)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = x
        if self.use_shortcut:
            shortcut = self.shortcut_bn(self.shortcut_conv(x))
        out += shortcut
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, symmetry='vanilla'):
        super(Bottleneck, self).__init__()
        self.use_shortcut = stride != 1 or in_planes != self.expansion*planes
        self.conv1 = SymmetricConv2d(in_planes, planes, kernel_size=1, bias=False, symmetry=symmetry)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = SymmetricConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, symmetry=symmetry)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.conv3 = SymmetricConv2d(planes, self.expansion*planes, kernel_size=1, bias=False, symmetry=symmetry)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, affine=True)
        
        self.shortcut_conv = nn.Sequential()
        if self.use_shortcut:
            self.shortcut_conv = SymmetricConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, symmetry=symmetry)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = x
        if self.use_shortcut:
            shortcut = self.shortcut_bn(self.shortcut_conv(x))
        out += shortcut
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cfg=None, symmetry='vanilla'):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = SymmetricConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, symmetry=symmetry)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1, symmetry=symmetry)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, symmetry=symmetry)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, symmetry=symmetry)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, symmetry=symmetry)
        self.output_dim = 512*block.expansion
        self.linear = nn.Linear(self.output_dim, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride, symmetry):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, symmetry=symmetry))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, use_linear=True):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        if use_linear:
            out = self.linear(out)
        return out
    
    def to_normal(self):
        normal_model = copy.deepcopy(self)
        replace_sym_conv_with_normal(normal_model)
        return normal_model

    def evaluate_symmetry(self):
        with torch.no_grad():
            symmetries = evaluate_symmetry(self)
        return symmetries
    

class ResNet_basic(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cfg=None, symmetry='vanilla'):
        super(ResNet_basic, self).__init__()

        self.in_planes = 16
        self.conv1 = SymmetricConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, symmetry=symmetry)
        self.bn1 = nn.BatchNorm2d(16, affine=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, symmetry=symmetry)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, symmetry=symmetry)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, symmetry=symmetry)
        self.output_dim = 64*block.expansion
        self.linear = nn.Linear(self.output_dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, symmetry):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, symmetry=symmetry))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, use_linear=True):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        if use_linear:
            out = self.linear(out)
        return out

def get_block(block):
    if block == "BasicBlock":
        return BasicBlock
    elif block == "Bottleneck":
        return Bottleneck

def ResNet18(num_classes=10, block="BasicBlock", symmetry='vanilla'):
    return ResNet(get_block(block), [2, 2, 2, 2], num_classes=num_classes, symmetry=symmetry)

def ResNet34(num_classes=10, block="BasicBlock", symmetry='vanilla'):
    return ResNet(get_block(block), [3, 4, 6, 3], num_classes=num_classes, symmetry=symmetry)

def ResNet56(num_classes=10, block="BasicBlock", symmetry='vanilla'):
    return ResNet_basic(get_block(block), [9, 9, 9], num_classes=num_classes, symmetry=symmetry)


def create_model(name, num_classes=10, block='BasicBlock', symmetry='vanilla', normal_conv_layer=True):
    if name == 'ResNet18':
        net = ResNet18(num_classes=num_classes, block=block, symmetry=symmetry)
    if normal_conv_layer:
        #turn the symmetric into a standard layer
        net = net.to_normal()
    return net
