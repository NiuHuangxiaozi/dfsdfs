# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Image classifiation.
"""
import mindspore.nn as nn


class Vgg(nn.Cell):
    """
    TO-DO.

    参数:
        num_classes (int): Class numbers. Default: 5.
        phase (int): 指定是训练/评估阶段

    返回值:
        Tensor, infer output tensor.

    example：
    	self.layer1_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,weight_init='XavierUniform')
        self.layer1_bn1 = nn.BatchNorm2d(num_features=64)
        self.layer1_relu1 = nn.LeakyReLU()

    """

    def __init__(self, num_classes=5, args=None, phase="train"):
        super(Vgg, self).__init__()
        dropout_ratio = 0.5
        if not args.has_dropout or phase == "test":
            dropout_ratio = 1.0

        # TO-DO:构建vgg17
        # layer1
        self.layer1_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.layer1_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.layer1_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer2
        self.layer2_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.layer2_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.layer2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer3
        self.layer3_conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.layer3_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.layer3_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.layer3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer4
        self.layer4_conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.layer4_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.layer4_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.layer4_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer5
        self.layer5_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.layer5_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.layer5_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.layer5_conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.layer5_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fullyconnect1 = nn.Dense(in_channels=25088, out_channels=4096)
        self.fullyconnect2 = nn.Dense(in_channels=4096, out_channels=4096)
        self.fullyconnect3 = nn.Dense(in_channels=4096, out_channels=num_classes)

        self.relu = nn.ReLU()
        self.BN1_1 = nn.BatchNorm2d(num_features=64)
        self.BN1_2 = nn.BatchNorm2d(num_features=64)
        self.BN2_1 = nn.BatchNorm2d(num_features=128)
        self.BN2_2 = nn.BatchNorm2d(num_features=128)
        self.BN3_1 = nn.BatchNorm2d(num_features=256)
        self.BN3_2 = nn.BatchNorm2d(num_features=256)
        self.BN3_3 = nn.BatchNorm2d(num_features=256)
        self.BN4_1 = nn.BatchNorm2d(num_features=512)
        self.BN4_2 = nn.BatchNorm2d(num_features=512)
        self.BN4_3 = nn.BatchNorm2d(num_features=512)
        self.BN5_1 = nn.BatchNorm2d(num_features=512)
        self.BN5_2 = nn.BatchNorm2d(num_features=512)
        self.BN5_3 = nn.BatchNorm2d(num_features=512)
        self.BN5_4 = nn.BatchNorm2d(num_features=512)

        self.dropout = nn.Dropout(dropout_ratio)
        self.leakrelu = nn.LeakyReLU()

    def construct(self, x):
        # layer1
        x = self.relu(self.BN1_1(self.layer1_conv1(x)))
        x = self.relu(self.BN1_2(self.layer1_conv2(x)))
        x = self.layer1_maxpool(x)

        # layer2
        x = self.relu(self.BN2_1(self.layer2_conv1(x)))
        x = self.relu(self.BN2_2(self.layer2_conv2(x)))
        x = self.layer2_maxpool(x)

        # layer3
        x = self.relu(self.BN3_1(self.layer3_conv1(x)))
        x = self.relu(self.BN3_2(self.layer3_conv2(x)))
        x = self.relu(self.BN3_3(self.layer3_conv3(x)))
        x = self.layer3_maxpool(x)

        # layer4
        x = self.relu(self.BN4_1(self.layer4_conv1(x)))
        x = self.relu(self.BN4_2(self.layer4_conv2(x)))
        x = self.relu(self.BN4_3(self.layer4_conv3(x)))
        x = self.layer4_maxpool(x)

        # layer5
        x = self.relu(self.BN5_1(self.layer5_conv1(x)))
        x = self.relu(self.BN5_2(self.layer5_conv2(x)))
        x = self.relu(self.BN5_3(self.layer5_conv3(x)))
        x = self.relu(self.BN5_4(self.layer5_conv4(x)))
        x = self.layer5_maxpool(x)

        # flatten
        x = self.flatten(x)

        # three fullyconnected network
        x = self.fullyconnect1(x)
        x = self.leakrelu(x)
        x = self.dropout(x)
        x = self.fullyconnect2(x)
        x = self.leakrelu(x)
        x = self.dropout(x)
        x = self.fullyconnect3(x)

        # x=nn.Softmax(axis=1)(x)
        return x


def vgg17(num_classes=1000, args=None, phase="train", **kwargs):
    """
    生成VGG17网络实例
    参数:
        num_classes (int): 分类数
        args (namespace): 参数
        phase (str): 指定是训练/评估阶段
    返回:
        Cell, cell instance of Vgg17 neural network with Batch Normalization.

    参考如下:
        >>> vgg17(num_classes=5, args=args, **kwargs)
    """
    net = Vgg(num_classes=num_classes, args=args, phase=phase, **kwargs)
    return net
