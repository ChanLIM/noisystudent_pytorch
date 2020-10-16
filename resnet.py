from torch import nn
from torch.nn import functional as F


__all__ = ['ResNet', 'ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet18_Dropout', 'ResNet34_Dropout', 'ResNet50_Dropout', 'ResNet101_Dropout', 'ResNet152_Dropout']

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_Dropout(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=10):
        super(ResNet_Dropout, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


class ResNet_2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_2, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

        self.apply(initialize_weights)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def ResNet18(in_channels=1, num_classes=10):
    return ResNet(
        BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes
    )


def ResNet34(in_channels=1, num_classes=10):
    return ResNet(
        BasicBlock, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes
    )


def ResNet50(in_channels=1, num_classes=10):
    return ResNet(
        Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes
    )


def ResNet101(in_channels=1, num_classes=10):
    return ResNet(
        Bottleneck, [3, 4, 23, 3], in_channels=in_channels, num_classes=num_classes
    )


def ResNet152(in_channels=1, num_classes=10):
    return ResNet(
        Bottleneck, [3, 8, 36, 3], in_channels=in_channels, num_classes=num_classes
    )



def ResNet18_Dropout(in_channels=1, num_classes=10):
    return ResNet_Dropout(
        BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes
    )


def ResNet34_Dropout(in_channels=1, num_classes=10):
    return ResNet_Dropout(
        BasicBlock, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes
    )


def ResNet50_Dropout(in_channels=1, num_classes=10):
    return ResNet_Dropout(
        Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes
    )


def ResNet101_Dropout(in_channels=1, num_classes=10):
    return ResNet_Dropout(
        Bottleneck, [3, 4, 23, 3], in_channels=in_channels, num_classes=num_classes
    )


def ResNet152_Dropout(in_channels=1, num_classes=10):
    return ResNet_Dropout(
        Bottleneck, [3, 8, 36, 3], in_channels=in_channels, num_classes=num_classes
    )



def ResNet20(in_channels=1, num_classes=10):
    return ResNet_2(
        Bottleneck, [3, 3, 3], num_classes=num_classes
    )


def ResNet32(in_channels=1, num_classes=10):
    return ResNet_2(
        Bottleneck, [5, 5, 5], num_classes=num_classes
    )


def ResNet44(in_channels=1, num_classes=10):
    return ResNet_2(
        Bottleneck, [7, 7, 7], num_classes=num_classes
    )


def ResNet56(in_channels=1, num_classes=10):
    return ResNet_2(
        Bottleneck, [9, 9, 9], num_classes=num_classes
    )


def ResNet110(in_channels=1, num_classes=10):
    return ResNet_2(
        Bottleneck, [18, 18, 18], num_classes=num_classes
    )