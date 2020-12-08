import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

__all__ = ['MobileNetV2', 'MobileNet_V2']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
}

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def ConvBNReLU(inp, oup, kernel_size=3, stride=1, groups=1):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
            nn.Conv2d(inp,
                      oup,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
            )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, dataset, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        self.dataset = dataset
        if dataset == 'imagenet':
            num_classes = 1000
        elif dataset == 'cub200':
            num_classes = 200
        elif dataset == 'cars196':
            num_classes = 196
        elif dataset == 'stanford':
            num_classes = 11318
        elif dataset == 'dogs120':
            num_classes = 120

        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1]
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        if dataset == 'imagenet':
            self.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(self.last_channel, num_classes)
            )
        else:
            self.classifier1 = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(self.last_channel, num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x) if self.dataset == 'imagenet' else self.classifier1(x)
        return x

def MobileNet_V2(dataset, pretrained=False, **kwargs):
    model = MobileNetV2(dataset=dataset, **kwargs)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['mobilenet_v2']), strict=False)
        print('MobileNet-V2 Use pretrained model for initialization')
    return model

if __name__ == '__main__':
    model = MobileNet_V2(pretrained=True, dataset='cub200', width_mult=1.0)
    print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    #model.extract_feature(x, True)
