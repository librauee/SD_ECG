import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR)




model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed1d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, bias=False, padding=3)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=11, stride=stride,
                               padding=5, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=7, bias=False, padding=3)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=18):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNetBig(nn.Module):

    def __init__(self, block, layers, num_classes=18):
        self.inplanes = 64
        super(ResNetBig, self).__init__()

        self.conv0 = nn.Conv1d(12, 32, kernel_size=31, stride=2, padding=15,
                               bias=False)
        self.bn0 = nn.BatchNorm1d(32)
        self.conv1 = nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetPlus(nn.Module):

    def __init__(self, block, layers, num_classes=18):
        self.inplanes = 64
        super(ResNetPlus, self).__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)

        #self.fc1 = nn.Linear

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.avgpool(x)
        x2 = self.maxpool2(x)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2), 1)
        x = self.fc(x)

        return x

class ResNetPlusAdd(nn.Module):

    def __init__(self, block, layers, num_classes=18):
        self.inplanes = 64
        super(ResNetPlusAdd, self).__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * block.expansion * 2 + 128, num_classes)

        #self.fc1 = nn.Linear

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feature):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.avgpool(x)
        x2 = self.maxpool2(x)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2, feature), 1)
        x = self.fc(x)

        return x

class ResNetPlus2(nn.Module):

    def __init__(self, block, layers, num_classes=18):
        self.inplanes = 64
        super(ResNetPlus2, self).__init__()

        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(12, 64, kernel_size=31, stride=2, padding=15,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(12, 64, kernel_size=45, stride=2, padding=22,
                               bias=False)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(12, 64, kernel_size=9, stride=2, padding=4,
                               bias=False)
        self.bn4 = nn.BatchNorm1d(64)


        self.conv11 = nn.Conv1d(64, 16, kernel_size=1, bias=False)


        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)

        #self.fc1 = nn.Linear

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv11(x1)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.conv11(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)

        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        x4 = self.conv11(x4)

        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.avgpool(x)
        x2 = self.maxpool2(x)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2), 1)
        x = self.fc(x)

        return x


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class ResNetPlus3(nn.Module):

    def __init__(self, block, layers, num_classes=18):
        self.inplanes = 64
        super(ResNetPlus3, self).__init__()

        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(12, 64, kernel_size=31, stride=2, padding=15,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(12, 64, kernel_size=45, stride=2, padding=22,
                               bias=False)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(12, 64, kernel_size=9, stride=2, padding=4,
                               bias=False)
        self.bn4 = nn.BatchNorm1d(64)


        self.conv11 = nn.Conv1d(64, 16, kernel_size=1, bias=False)
        self.se = SELayer(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(512 * block.expansion * 2, 128),
            Hswish(inplace=True),
            nn.Linear(128, num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv11(x1)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.conv11(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)

        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        x4 = self.conv11(x4)

        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.se(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.avgpool(x)
        x2 = self.maxpool2(x)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2), 1)
        x = self.head(x)

        return x


class ResNet_(nn.Module):

    def __init__(self, block, layers, num_classes=18):
        self.inplanes = 64
        super(ResNet_, self).__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc(x)
        x2 = self.fc2(x)

        return x1, x2


class RNN(nn.Module):
    """RNN module(cell type lstm or gru)"""

    def __init__(
            self,
            input_size,
            hid_size,
            num_rnn_layers=1,
            dropout_p=0.2,
            bidirectional=False,
            rnn_type='lstm',
    ):
        super().__init__()

        if rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hid_size,
                num_layers=num_rnn_layers,
                dropout=dropout_p if num_rnn_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )

        else:
            self.rnn_layer = nn.GRU(
                input_size=input_size,
                hidden_size=hid_size,
                num_layers=num_rnn_layers,
                dropout=dropout_p if num_rnn_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )

    def forward(self, input):
        outputs, hidden_states = self.rnn_layer(input)
        return outputs, hidden_states

class ResNet_LSTM(nn.Module):

    def __init__(self, block, layers, num_classes=18):
        self.inplanes = 64
        super(ResNet_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.rnn_layer = RNN(
            input_size=209,
            hid_size=256,
            rnn_type='lstm',
        )
        # self.attn = nn.Linear(256, 256, bias=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        x_out, hid_states = self.rnn_layer(x)
        # print(x_out.shape)
        # print(hid_states[0].shape)
        # print(hid_states[1].shape)
        # x = torch.cat([hid_states[0], hid_states[1]], dim=0).transpose(0, 1)
        # print(x.shape)
        # x_attn = torch.tanh(self.attn(x))
        # print(x_attn.shape)
        # x = x_attn.bmm(x_out)
        # x = x.transpose(2, 1)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1) * x.size(2))

        x = self.fc(x)

        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def se_resnet18(pretrained=False, **kwargs):

    model = ResNet(SEBasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet50_plus():
    return ResNetPlus(Bottleneck, [3, 4, 6, 3])


def se_resnet34(pretrained=False, **kwargs):

    model = ResNet(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def se_resnet34_plus(pretrained=False, **kwargs):

    model = ResNetPlus(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def se_resnet34_plus_add(pretrained=False, **kwargs):

    model = ResNetPlusAdd(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def se_resnet34_plus2(pretrained=False, **kwargs):

    model = ResNetPlus2(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def se_resnet34_plus3(pretrained=False, **kwargs):

    model = ResNetPlus3(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def se_resnet34_big(pretrained=False, **kwargs):

    model = ResNetBig(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def multi_se_resnet34(pretrained=False, **kwargs):

    model = ResNet_(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def se_resnet34_lstm(pretrained=False, **kwargs):

    model = ResNet_LSTM(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet50(pretrained=False, **kwargs):

    model = ResNet(SEBottleneck, [3, 4, 6, 3], **kwargs)
    return model


import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import torch


def conv_2d(in_planes, out_planes, stride=(1, 1), size=3):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, size), stride=stride,
                     padding=(0, (size - 1) // 2), bias=False)


def conv_1d(in_planes, out_planes, stride=1, size=3):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=size, stride=stride,
                     padding=(size - 1) // 2, bias=False)


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, size=3, res=True):
        super(BasicBlock1d, self).__init__()
        self.conv1 = conv_1d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_1d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv_1d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout(.2)
        self.downsample = downsample
        self.stride = stride
        self.res = res

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
        # out = self.relu(out)

        return out


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None, size=3, res=True):
        super(BasicBlock2d, self).__init__()
        self.conv1 = conv_2d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_2d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_2d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(.2)
        self.downsample = downsample
        self.stride = stride
        self.res = res

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
        # out = self.relu(out)

        return out


class ECGNet(nn.Module):
    def __init__(self, input_channel=1, num_classes=18):  # , layers=[2, 2, 2, 2, 2, 2]
        sizes = [
            [3, 3, 3, 3, 3, 3],
            [5, 5, 5, 5, 3, 3],
            [7, 7, 7, 7, 3, 3],
        ]
        self.sizes = sizes
        layers = [
            [3, 3, 2, 2, 2, 2],
            [3, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ]

        super(ECGNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(1, 50), stride=(1, 2), padding=(0, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 16), stride=(1, 2), padding=(0, 0),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=(1,16), stride=(1,2), padding=(0,0),
        #                       bias=False)
        # print(self.conv2)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.dropout = nn.Dropout(.2)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 0))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.layers1_list = nn.ModuleList()
        self.layers2_list = nn.ModuleList()
        for i, size in enumerate(sizes):
            self.inplanes = 32
            self.layers1 = nn.Sequential()
            self.layers2 = nn.Sequential()
            self.layers1.add_module('layer{}_1_1'.format(size),
                                    self._make_layer2d(BasicBlock2d, 32, layers[i][0], stride=(1, 1), size=sizes[i][0]))
            self.layers1.add_module('layer{}_1_2'.format(size),
                                    self._make_layer2d(BasicBlock2d, 32, layers[i][1], stride=(1, 1), size=sizes[i][1]))
            self.inplanes *= 12
            self.layers2.add_module('layer{}_2_1'.format(size),
                                    self._make_layer1d(BasicBlock1d, 384, layers[i][2], stride=2, size=sizes[i][2]))
            self.layers2.add_module('layer{}_2_2'.format(size),
                                    self._make_layer1d(BasicBlock1d, 384, layers[i][3], stride=2, size=sizes[i][3]))
            self.layers2.add_module('layer{}_2_3'.format(size),
                                    self._make_layer1d(BasicBlock1d, 384, layers[i][4], stride=2, size=sizes[i][4]))
            self.layers2.add_module('layer{}_2_4'.format(size),
                                    self._make_layer1d(BasicBlock1d, 384, layers[i][5], stride=2, size=sizes[i][5]))

            self.layers1_list.append(self.layers1)
            self.layers2_list.append(self.layers2)

        # self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(384 * len(sizes), num_classes)

    def _make_layer1d(self, block, planes, blocks, stride=2, size=3, res=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))

        return nn.Sequential(*layers)

    def _make_layer2d(self, block, planes, blocks, stride=(1, 2), size=3, res=True):
        downsample = None
        if stride != (1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=(1, 1), padding=(0, 0), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = x0.unsqueeze(1)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        x0 = self.conv2(x0)
        # x0 = self.bn2(x0)
        # x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        # x0 = self.dropout(x0)

        xs = []
        for i in range(len(self.sizes)):
            # print(self.layers1_list[i])
            x = self.layers1_list[i](x0)
            x = torch.flatten(x, start_dim=1, end_dim=2)
            x = self.layers2_list[i](x)
            x = self.avgpool(x)
            xs.append(x)
        out = torch.cat(xs, dim=2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


# class Bottleneck(nn.Module):
#     def __init__(self, in_planes, growth_rate):
#         super(Bottleneck, self).__init__()
#         self.bn1 = nn.BatchNorm1d(in_planes)
#         self.conv1 = nn.Conv1d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(4*growth_rate)
#         self.conv2 = nn.Conv1d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
#
#     def forward(self, x):
#         out = self.conv1(F.relu(self.bn1(x)))
#         out = self.conv2(F.relu(self.bn2(out)))
#         out = torch.cat([out,x], 1)
#         return out
#
#
# class Transition(nn.Module):
#     def __init__(self, in_planes, out_planes):
#         super(Transition, self).__init__()
#         self.bn = nn.BatchNorm1d(in_planes)
#         self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#
#     def forward(self, x):
#         out = self.conv(F.relu(self.bn(x)))
#         out = self.avg_pool(out)
#         return out
#
#
# class DenseNet(nn.Module):
#     def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=18):
#         super(DenseNet, self).__init__()
#         self.growth_rate = growth_rate
#
#         num_planes = 2*growth_rate
#         self.conv1 = nn.Conv1d(12, num_planes, kernel_size=3, padding=1, bias=False)
#
#         self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
#         num_planes += nblocks[0]*growth_rate
#         out_planes = int(math.floor(num_planes*reduction))
#         self.trans1 = Transition(num_planes, out_planes)
#         num_planes = out_planes
#
#         self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
#         num_planes += nblocks[1]*growth_rate
#         out_planes = int(math.floor(num_planes*reduction))
#         self.trans2 = Transition(num_planes, out_planes)
#         num_planes = out_planes
#
#         self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
#         num_planes += nblocks[2]*growth_rate
#         out_planes = int(math.floor(num_planes*reduction))
#         self.trans3 = Transition(num_planes, out_planes)
#         num_planes = out_planes
#
#         self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
#         num_planes += nblocks[3]*growth_rate
#
#         self.bn = nn.BatchNorm1d(num_planes)
#         self.linear = nn.Linear(num_planes, num_classes)
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#
#     def _make_dense_layers(self, block, in_planes, nblock):
#         layers = []
#         for i in range(nblock):
#             layers.append(block(in_planes, self.growth_rate))
#             in_planes += self.growth_rate
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.trans1(self.dense1(out))
#         out = self.trans2(self.dense2(out))
#         out = self.trans3(self.dense3(out))
#         out = self.dense4(out)
#         out = self.avg_pool(F.relu(self.bn(out)))
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
#
# def densenet121():
#     return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import copy
# import math
#
# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model, dropout=0.1, max_len=16001):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term) # 偶数番目
#         pe[:, 1::2] = torch.cos(position * div_term) # 奇数番目
#         pe = pe.unsqueeze(0).transpose(0, 1)
#
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         # return self.dropout(x)
#         return x
#
# def _get_activation_fn(activation):
#     if activation == "relu":
#         return F.relu
#     elif activation == "gelu":
#         return F.gelu
#
#     raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
#
# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
#         super(TransformerEncoderLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#
#         self.activation = _get_activation_fn(activation)
#
#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(TransformerEncoderLayer, self).__setstate__(state)
#
#     def forward(self, src, src_mask=None, src_key_padding_mask=None):
#         # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
#         src2, weights = self.self_attn(src, src, src, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src, weights
#
#
# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
#
# class TransformerEncoder(nn.Module):
#     __constants__ = ['norm']
#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super(TransformerEncoder, self).__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#
#     def forward(self, src, mask=None, src_key_padding_mask=None):
#         # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
#         output = src
#         weights = []
#         for mod in self.layers:
#             output, weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
#             weights.append(weight)
#
#         if self.norm is not None:
#             output = self.norm(output)
#         return output, weights
#
#
# class ECGNetTrans(nn.Module):
#     def __init__(self, in_channel=12, ninp=512, nhead=4, nhid=1024, dropout=0.1, nlayers=2):
#         super(ECGNetTrans, self).__init__()
#
#         d_model = 256
#         encoder_layers = TransformerEncoderLayer(d_model, nhead, nhid, dropout)
#         self.transformer = TransformerEncoder(encoder_layers, nlayers)
#
#
#         self.top_layer1 = nn.Sequential(
#                              nn.BatchNorm1d(12),
#                              nn.ReLU(inplace=True),
#                              nn.Conv1d(in_channel, 32, kernel_size=50, stride=2),
#                          )
#
#         self.top_layer2 = nn.Sequential(
#                              nn.BatchNorm1d(32),
#                              nn.ReLU(inplace=True),
#                              nn.Conv1d(32, 64, kernel_size=15, stride=2),
#                          )
#
#         self.top_layer3 = nn.Sequential(
#                              nn.BatchNorm1d(64),
#                              nn.ReLU(inplace=True),
#                              nn.Conv1d(64, 128, kernel_size=15, stride=2),
#                          )
#
#         self.top_layer4 = nn.Sequential(
#                              nn.BatchNorm1d(128),
#                              nn.ReLU(inplace=True),
#                              nn.Conv1d(128, 256, kernel_size=15, stride=2),
#         )
#
#
#         self.bottom_linear = nn.Sequential(
#                                  nn.Linear(d_model, d_model//2),
#                                  nn.ReLU(inplace=True),
#                                  nn.Dropout(dropout),
#                                  nn.Linear(d_model//2, 18)
#                              )
#
#         self.l = nn.Linear(d_model, 18)
#
#         self.pos_encoder = PositionalEncoding(d_model)
#
#         self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model//2, num_layers=1,
#                                     batch_first=False, bidirectional=True)
#
#
#
#     def forward(self, x, info=None):
#         x = x.squeeze(1) # torch.Size([1, 12, 5000])
#         x = self.top_layer1(x) # torch.Size([1, 32, 4988])  -> [1, 32, 2476] ks=50, s=2
#         x = self.top_layer2(x) # torch.Size([1, 64, 4982])  -> [1, 64, 1231] ks=15, s=2
#         x = self.top_layer3(x) # torch.Size([1, 128, 4978]) -> [1, 128, 609] ks=15, s=2
#         x = self.top_layer4(x) # torch.Size([1, 256, 4976]) -> [1, 256, 298] ks=15, s=2
#         x = x.permute(2, 0, 1) # torch.Size([4976, 1, 256]) -> [298, 1, 256]
#
#         # x = self.pos_encoder(x) # torch.Size([4976, 1, 256])      -> [298, 1, 256]
#         x, _ = self.lstm(x)
#         x_t, _ = self.transformer(x) # torch.Size([4976, 1, 256]) -> [298, 1, 256]
#
#         x = x_t.permute(1, 2, 0) # torch.Size([1, 256, 4976])                   -> [1, 256, 298]
#         x = F.max_pool1d(x, kernel_size=x.size()[2:]) # torch.Size([1, 256, 1]) -> [1, 256, 1]
#         x = x.contiguous().view(x.size()[0], -1) # torch.Size([1, 256])
#         if info != None:
#             info = torch.unsqueeze(info, 1)
#             x = torch.cat([x, info], dim=1)
#
#         # x = self.bottom_linear(x)
#         x = self.l(x)
#         return x
#
#
# class ResBlock(nn.Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsample=None):
#         super(ResBlock, self).__init__()
#         self.bn1 = nn.BatchNorm1d(num_features=in_channels)
#         self.relu = nn.ReLU(inplace=False)
#         self.dropout = nn.Dropout(p=0.1, inplace=False)
#         self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                stride=stride, padding=padding, bias=False)
#         self.bn2 = nn.BatchNorm1d(num_features=out_channels)
#         self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                stride=stride, padding=padding, bias=False)
#         self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.conv1(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.conv2(out)
#
#         if self.downsample is not None:
#             out = self.maxpool(out)
#             identity = self.downsample(x)
#
#         out += identity
#         # print(out.shape)
#
#         return out
#
#
# class ECGNet(nn.Module):
#
#     def __init__(self, struct=[15, 17, 19, 21], in_channels=12, fixed_kernel_size=17, num_classes=18):
#         super(ECGNet, self).__init__()
#         self.struct = struct
#         self.planes = 16
#         self.parallel_conv = nn.ModuleList()
#
#         for i, kernel_size in enumerate(struct):
#             sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=kernel_size,
#                                  stride=1, padding=0, bias=False)
#             self.parallel_conv.append(sep_conv)
#         # self.parallel_conv.append(nn.Sequential(
#         #     nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
#         #     nn.Conv1d(in_channels=1, out_channels=self.planes, kernel_size=1,
#         #                        stride=1, padding=0, bias=False)
#         # ))
#
#         self.bn1 = nn.BatchNorm1d(num_features=self.planes)
#         self.relu = nn.ReLU(inplace=False)
#         self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size,
#                                stride=2, padding=2, bias=False)
#         self.block = self._make_layer(kernel_size=fixed_kernel_size, stride=1, padding=8)
#         self.bn2 = nn.BatchNorm1d(num_features=self.planes)
#         self.avgpool = nn.AvgPool1d(kernel_size=8, stride=8, padding=2)
#         self.rnn = nn.LSTM(input_size=12, hidden_size=40, num_layers=1, bidirectional=False)
#         self.fc = nn.Linear(in_features=2024, out_features=num_classes)
#
#     def _make_layer(self, kernel_size, stride, blocks=15, padding=0):
#         layers = []
#         downsample = None
#         base_width = self.planes
#
#         for i in range(blocks):
#             if (i + 1) % 4 == 0:
#                 downsample = nn.Sequential(
#                     nn.Conv1d(in_channels=self.planes, out_channels=self.planes + base_width, kernel_size=1,
#                               stride=1, padding=0, bias=False),
#                     nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#                 )
#                 layers.append(
#                     ResBlock(in_channels=self.planes, out_channels=self.planes + base_width, kernel_size=kernel_size,
#                              stride=stride, padding=padding, downsample=downsample))
#                 self.planes += base_width
#             elif (i + 1) % 2 == 0:
#                 downsample = nn.Sequential(
#                     nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#                 )
#                 layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
#                                        stride=stride, padding=padding, downsample=downsample))
#             else:
#                 downsample = None
#                 layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
#                                        stride=stride, padding=padding, downsample=downsample))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x, info=None):
#         out_sep = []
#
#         for i in range(len(self.struct)):
#             sep = self.parallel_conv[i](x)
#             out_sep.append(sep)
#
#         out = torch.cat(out_sep, dim=2)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv1(out)  # out => [b, 16, 9960]
#
#         out = self.block(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.avgpool(out)  # out => [b, 64, 10]
#         out = out.reshape(out.shape[0], -1)  # out => [b, 640]
#
#         rnn_out, (rnn_h, rnn_c) = self.rnn(x.permute(2, 0, 1))
#         new_rnn_h = rnn_h[-1, :, :]  # rnn_h => [b, 40]
#
#         new_out = torch.cat([out, new_rnn_h], dim=1)  # out => [b, 680]
#         # print(new_out.shape)
#         if info != None:
#             info = torch.unsqueeze(info, 1)
#             # print(info.shape)
#             new_out = torch.cat([new_out, info], dim=1)
#             # print('concate extra feature!!!')
#         result = self.fc(new_out)  # out => [b, 2]
#
#         # print(out.shape)
#
#         return result
#
#
# class MyConv1dPadSame(nn.Module):
#     """
#     extend nn.Conv1d to support SAME padding
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
#         super(MyConv1dPadSame, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.groups = groups
#         self.conv = torch.nn.Conv1d(
#             in_channels=self.in_channels,
#             out_channels=self.out_channels,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             groups=self.groups)
#
#     def forward(self, x):
#         net = x
#
#         # compute pad shape
#         in_dim = net.shape[-1]
#         out_dim = (in_dim + self.stride - 1) // self.stride
#         p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
#         pad_left = p // 2
#         pad_right = p - pad_left
#         net = F.pad(net, (pad_left, pad_right), "constant", 0)
#
#         net = self.conv(net)
#
#         return net
#
#
# class MyMaxPool1dPadSame(nn.Module):
#     """
#     extend nn.MaxPool1d to support SAME padding
#     """
#
#     def __init__(self, kernel_size):
#         super(MyMaxPool1dPadSame, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = 1
#         self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)
#
#     def forward(self, x):
#         net = x
#
#         # compute pad shape
#         in_dim = net.shape[-1]
#         out_dim = (in_dim + self.stride - 1) // self.stride
#         p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
#         pad_left = p // 2
#         pad_right = p - pad_left
#         net = F.pad(net, (pad_left, pad_right), "constant", 0)
#
#         net = self.max_pool(net)
#
#         return net
#
#
# class BasicBlock(nn.Module):
#     """
#     ResNet Basic Block
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do,
#                  is_first_block=False):
#         super(BasicBlock, self).__init__()
#
#         self.in_channels = in_channels
#         self.kernel_size = kernel_size
#         self.out_channels = out_channels
#         self.stride = stride
#         self.groups = groups
#         self.downsample = downsample
#         if self.downsample:
#             self.stride = stride
#         else:
#             self.stride = 1
#         self.is_first_block = is_first_block
#         self.use_bn = use_bn
#         self.use_do = use_do
#
#         # the first conv
#         self.bn1 = nn.BatchNorm1d(in_channels)
#         self.relu1 = nn.ReLU()
#         self.do1 = nn.Dropout(p=0.5)
#         self.conv1 = MyConv1dPadSame(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=self.stride,
#             groups=self.groups)
#
#         # the second conv
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.relu2 = nn.ReLU()
#         self.do2 = nn.Dropout(p=0.5)
#         self.conv2 = MyConv1dPadSame(
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=1,
#             groups=self.groups)
#
#         self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)
#
#     def forward(self, x):
#
#         identity = x
#
#         # the first conv
#         out = x
#         if not self.is_first_block:
#             if self.use_bn:
#                 out = self.bn1(out)
#             out = self.relu1(out)
#             if self.use_do:
#                 out = self.do1(out)
#         out = self.conv1(out)
#
#         # the second conv
#         if self.use_bn:
#             out = self.bn2(out)
#         out = self.relu2(out)
#         if self.use_do:
#             out = self.do2(out)
#         out = self.conv2(out)
#
#         # if downsample, also downsample identity
#         if self.downsample:
#             identity = self.max_pool(identity)
#
#         # if expand channel, also pad zeros to identity
#         if self.out_channels != self.in_channels:
#             identity = identity.transpose(-1, -2)
#             ch1 = (self.out_channels - self.in_channels) // 2
#             ch2 = self.out_channels - self.in_channels - ch1
#             identity = F.pad(identity, (ch1, ch2), "constant", 0)
#             identity = identity.transpose(-1, -2)
#
#         # shortcut
#         out += identity
#
#         return out
#
#
# class ResNet1D(nn.Module):
#     """
#
#     Input:
#         X: (n_samples, n_channel, n_length)
#         Y: (n_samples)
#
#     Output:
#         out: (n_samples)
#
#     Pararmetes:
#         in_channels: dim of input, the same as n_channel
#         base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
#         kernel_size: width of kernel
#         stride: stride of kernel moving
#         groups: set larget to 1 as ResNeXt
#         n_block: number of blocks
#         n_classes: number of classes
#
#     """
#
#     def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2,
#                  increasefilter_gap=4, use_bn=True, use_do=True, verbose=False):
#         super(ResNet1D, self).__init__()
#
#         self.verbose = verbose
#         self.n_block = n_block
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.groups = groups
#         self.use_bn = use_bn
#         self.use_do = use_do
#
#         self.downsample_gap = downsample_gap  # 2 for base model
#         self.increasefilter_gap = increasefilter_gap  # 4 for base model
#
#         # first block
#         self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters,
#                                                 kernel_size=self.kernel_size, stride=1)
#         self.first_block_bn = nn.BatchNorm1d(base_filters)
#         self.first_block_relu = nn.ReLU()
#         out_channels = base_filters
#
#         # residual blocks
#         self.basicblock_list = nn.ModuleList()
#         for i_block in range(self.n_block):
#             # is_first_block
#             if i_block == 0:
#                 is_first_block = True
#             else:
#                 is_first_block = False
#             # downsample at every self.downsample_gap blocks
#             if i_block % self.downsample_gap == 1:
#                 downsample = True
#             else:
#                 downsample = False
#             # in_channels and out_channels
#             if is_first_block:
#                 in_channels = base_filters
#                 out_channels = in_channels
#             else:
#                 # increase filters at every self.increasefilter_gap blocks
#                 in_channels = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
#                 if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
#                     out_channels = in_channels * 2
#                 else:
#                     out_channels = in_channels
#
#             tmp_block = BasicBlock(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=self.kernel_size,
#                 stride=self.stride,
#                 groups=self.groups,
#                 downsample=downsample,
#                 use_bn=self.use_bn,
#                 use_do=self.use_do,
#                 is_first_block=is_first_block)
#             self.basicblock_list.append(tmp_block)
#
#         # final prediction
#         self.final_bn = nn.BatchNorm1d(out_channels)
#         self.final_relu = nn.ReLU(inplace=True)
#         # self.do = nn.Dropout(p=0.5)
#         self.dense = nn.Linear(out_channels, n_classes)
#         # self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#
#         out = x
#
#         # first conv
#         if self.verbose:
#             print('input shape', out.shape)
#         out = self.first_block_conv(out)
#         if self.verbose:
#             print('after first conv', out.shape)
#         if self.use_bn:
#             out = self.first_block_bn(out)
#         out = self.first_block_relu(out)
#
#         # residual blocks, every block has two conv
#         for i_block in range(self.n_block):
#             net = self.basicblock_list[i_block]
#             if self.verbose:
#                 print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block,
#                                                                                                   net.in_channels,
#                                                                                                   net.out_channels,
#                                                                                                   net.downsample))
#             out = net(out)
#             if self.verbose:
#                 print(out.shape)
#
#         # final prediction
#         if self.use_bn:
#             out = self.final_bn(out)
#         out = self.final_relu(out)
#         out = out.mean(-1)
#         if self.verbose:
#             print('final pooling', out.shape)
#         # out = self.do(out)
#         out = self.dense(out)
#         if self.verbose:
#             print('dense', out.shape)
#         # out = self.softmax(out)
#         if self.verbose:
#             print('softmax', out.shape)
#
#         return out
#
#
# def res1D(pretrained=False, **kwargs):
#
#     model = ResNet1D(
#         in_channels=12,
#         base_filters=256,
#         n_block=4,
#         kernel_size=16,
#         stride=2,
#         groups=32,
#         verbose=False,
#         n_classes=18)
#     return model


# import torch.nn as nn
# import math
# import torch.utils.model_zoo as model_zoo
# import torch
# 
#
# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv1d(in_planes, out_planes,
#                      kernel_size=7,
#                      stride=stride,
#                      padding=3,
#                      bias=False)
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.downsample = downsample
#         self.stride = stride
#         self.dropout = nn.Dropout(.2)
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv1d(inplanes, planes,
#                                kernel_size=7,
#                                bias=False,
#                                padding=3)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.conv2 = nn.Conv1d(planes, planes,
#                                kernel_size=11,
#                                stride=stride,
#                                padding=5,
#                                bias=False)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.conv3 = nn.Conv1d(planes, planes * 4,
#                                kernel_size=7,
#                                bias=False,
#                                padding=3)
#         self.bn3 = nn.BatchNorm1d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.dropout = nn.Dropout(.2)
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.conv3(out)
#         out = self.bn3(out)
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=18):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7,
#                                bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.globavepool = nn.AdaptiveAvgPool1d(1)
#         self.globamaxpool = nn.AdaptiveMaxPool1d(1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         self.fc1 = nn.Linear(512, 256)
#         self.fc_atten_1 = nn.Linear(512, 512)
#         # self.fc_atten_2 = nn.Linear(128, 128)
#         # self.fc_atten_3 = nn.Linear(256, 256)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv1d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def attention(self, x, i):
#         out = torch.transpose(x, 1, 2)
#         if i == 0:
#             out = self.fc_atten_1(out)
#         elif i == 1:
#             out = self.fc_atten_2(out)
#         elif i == 2:
#             out = self.fc_atten_3(out)
#
#         out = torch.transpose(out, 1, 2)
#         weight = self.globavepool(out)
#         out = weight * x
#         return out
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         # x = self.attention(x, 0)
#         x = self.layer2(x)
#         # x = self.attention(x, 1)
#         x = self.layer3(x)
#         # x = self.attention(x, 2)
#         x = self.layer4(x)
#
#         x = self.attention(x, 0)
#         x = torch.transpose(x, 1, 2)
#         x = self.fc1(x)
#         x = torch.transpose(x, 1, 2)
#
#         x1 = self.avgpool(x)
#         x2 = self.globamaxpool(x)
#         x1 = x1.view(x1.size(0), -1)
#         x2 = x2.view(x2.size(0), -1)
#         x2 = torch.cat((x1, x2), 1)
#         #        print(x2.size(1))
#         x = self.fc(x2)
#
#         return x
#
#
# def se_resnet34_plus(pretrained=False, **kwargs):
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     return model


class res_lstm(nn.Module):
    def __init__(self, input_size=12):
        dim_hidden=64
        hidden = [256,128]
        super().__init__()
        self.conv=nn.Sequential(
            #------------------stage 1--------------------
            nn.Conv1d(input_size,dim_hidden,24,stride=3),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(),
            nn.Conv1d(dim_hidden,dim_hidden*2,24,stride=1),
            nn.BatchNorm1d(dim_hidden*2),
            nn.LeakyReLU(),
            #
            nn.MaxPool1d(3,stride=2),
            nn.Dropout(0.25),
            #------------------stage 2--------------------
            nn.Conv1d(dim_hidden*2,dim_hidden*3,12,stride=2),
            nn.BatchNorm1d(dim_hidden*3),
            nn.LeakyReLU(),
            nn.Conv1d(dim_hidden*3,dim_hidden*3,12,stride=1),
            nn.BatchNorm1d(dim_hidden*3),
            nn.LeakyReLU(),
            #
            nn.MaxPool1d(3,stride=2),
            nn.Dropout(0.25),
            #------------------stage 3--------------------
            nn.Conv1d(dim_hidden*3,dim_hidden*3,7,stride=2),
            nn.BatchNorm1d(dim_hidden*3),
            nn.LeakyReLU(),
            nn.Conv1d(dim_hidden*3,dim_hidden*3,7,stride=1),
            nn.BatchNorm1d(dim_hidden*3),
            nn.LeakyReLU(),
            #
            nn.MaxPool1d(3,stride=2),
            nn.Dropout(0.25),
            #------------------stage 4--------------------
            nn.Conv1d(dim_hidden*3,dim_hidden*4,5,stride=1),
            nn.BatchNorm1d(dim_hidden*4),
            nn.LeakyReLU(),
            nn.Conv1d(dim_hidden*4,dim_hidden*4,5,stride=1),
            nn.BatchNorm1d(dim_hidden*4),
            nn.LeakyReLU(),
            #
            nn.MaxPool1d(2,stride=1),
            nn.Dropout(0.25),
            )
        self.lstm1 =nn.LSTM(dim_hidden*4, hidden[0],
                            batch_first=True, bidirectional=True)
        self.lstm2=nn.LSTM(2 * hidden[0], hidden[1],
                            batch_first=True, bidirectional=True)
        self.head=nn.Sequential(
            nn.Linear(2 * hidden[1], 64),
            nn.SELU(),
            nn.Linear(64, 18)
            )
        self.dropout = nn.Dropout(0.2)
    #
    def attention_net(self, x, query, mask=None):

        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)

        p_attn = F.softmax(scores, dim = -1)
        context = torch.matmul(p_attn, x).sum(1)
        return context, p_attn
    def forward(self, x):
        x=self.conv(x)
        #print(x.shape)
        x=x.permute(0,2,1)
        x,_=self.lstm1(x)
        x,_=self.lstm2(x)
        query = self.dropout(x)
        x, _ = self.attention_net(x, query)
        x = self.head(x)
        return x