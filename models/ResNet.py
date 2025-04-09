'''
File copied form the MMC github repo and slighlty modified to support symmetric kernels
https://github.com/EkdeepSLubana/MMC
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch


##############################################
# Symmetry Operators and Symmetric Conv2d
##############################################

class WeightSymmetry(nn.Module):
    def forward(self, weight):
        return weight

class HorizontalFlipSymmetry(WeightSymmetry):
    def forward(self, weight):
        return 0.5 * (weight + torch.flip(weight, dims=[3]))

class VerticalFlipSymmetry(WeightSymmetry):
    def forward(self, weight):
        return 0.5 * (weight + torch.flip(weight, dims=[2]))

class HVFlipSymmetry(WeightSymmetry):
    def forward(self, weight):
        return 0.25 * (weight +
                       torch.flip(weight, dims=[2]) +
                       torch.flip(weight, dims=[3]) +
                       torch.flip(weight, dims=[2, 3]))

class Rot90Symmetry(WeightSymmetry):
    def forward(self, weight):
        rot90 = lambda w, k: torch.rot90(w, k=k, dims=[2, 3])
        return 0.25 * (weight + rot90(weight, 1) + rot90(weight, 2) + rot90(weight, 3))

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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias, **kwargs)
        self.symmetry = SYMMETRY_CLASSES[symmetry]()
        
    def forward(self, x):
        # Apply the symmetry operator to the convolution weight
        weight = self.symmetry(self.conv.weight)
        return F.conv2d(x, weight, self.conv.bias,
                        stride=self.conv.stride, padding=self.conv.padding,
                        dilation=self.conv.dilation, groups=self.conv.groups)
    
### Conversion 
def replace_sym_conv_with_normal(module):
    """
    Recursively replace all instances of SymmetricConv2d within the module
    with a regular nn.Conv2d whose weights are set to the effective symmetric weight.
    """
    for name, child in list(module._modules.items()):
        # Check if this module is one of our symmetric convolutions.
        if isinstance(child, SymmetricConv2d):
            conv_orig = child.conv  # The underlying convolution.
            # Create a new standard convolution with identical parameters.
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
            # The effective weight is what the forward pass uses:
            # effective = symmetry(conv_orig.weight)
            effective_weight = child.symmetry(conv_orig.weight).detach().clone()
            new_conv.weight.data.copy_(effective_weight)
            if conv_orig.bias is not None:
                new_conv.bias.data.copy_(conv_orig.bias.data)
            # Replace the symmetric module in the parent's dict.
            module._modules[name] = new_conv
        else:
            # Recurse into children.
            replace_sym_conv_with_normal(child)

##############################################
# Modified ResNet Blocks with Symmetric Kernels
##############################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, symmetry='vanilla'):
        super(BasicBlock, self).__init__()
        self.use_shortcut = (stride != 1 or in_planes != self.expansion * planes)
        
        self.conv1 = SymmetricConv2d(in_planes, planes, kernel_size=3, stride=stride,
                                     padding=1, bias=False, symmetry=symmetry)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = SymmetricConv2d(planes, planes, kernel_size=3, stride=1,
                                     padding=1, bias=False, symmetry=symmetry)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        
        if self.use_shortcut:
            self.shortcut_conv = SymmetricConv2d(in_planes, self.expansion * planes,
                                                  kernel_size=1, stride=stride, bias=False, symmetry=symmetry)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion * planes, affine=True)

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
        self.use_shortcut = (stride != 1 or in_planes != self.expansion * planes)
        
        self.conv1 = SymmetricConv2d(in_planes, planes, kernel_size=1, bias=False, symmetry=symmetry)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = SymmetricConv2d(planes, planes, kernel_size=3, stride=stride, padding=1,
                                     bias=False, symmetry=symmetry)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.conv3 = SymmetricConv2d(planes, self.expansion * planes, kernel_size=1, bias=False, symmetry=symmetry)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=True)
        
        if self.use_shortcut:
            self.shortcut_conv = SymmetricConv2d(in_planes, self.expansion * planes,
                                                  kernel_size=1, stride=stride, bias=False, symmetry=symmetry)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion * planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = x
        if self.use_shortcut:
            shortcut = self.shortcut_bn(self.shortcut_conv(x))
        out += shortcut
        return F.relu(out)

##############################################
# Modified ResNet Model with Symmetric Kernels
##############################################

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cfg=None, symmetry='vanilla'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # Use symmetric convolution for the initial layer as well.
        self.conv1 = SymmetricConv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                     bias=False, symmetry=symmetry)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1, symmetry=symmetry)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, symmetry=symmetry)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, symmetry=symmetry)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, symmetry=symmetry)
        
        self.output_dim = 512 * block.expansion
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
        return replace_sym_conv_with_normal(normal_model)

class ResNet_basic(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cfg=None, symmetry='vanilla'):
        super(ResNet_basic, self).__init__()
        self.in_planes = 16
        self.conv1 = SymmetricConv2d(3, 16, kernel_size=3, stride=1, padding=1,
                                     bias=False, symmetry=symmetry)
        self.bn1 = nn.BatchNorm2d(16, affine=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, symmetry=symmetry)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, symmetry=symmetry)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, symmetry=symmetry)
        self.output_dim = 64 * block.expansion
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

##############################################
# Convenience Functions
##############################################

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

##############################################
# Retrieval Function for Backbones
##############################################

def create_model(name, num_classes=10, block='BasicBlock', symmetry='vanilla', normal_conv_layer=False):
    if name == 'res18':
        net = ResNet18(num_classes=num_classes, block=block, symmetry=symmetry)
    if normal_conv_layer:
        #turn the symmetric into a standard layer
        net = net.to_normal()
    return net
