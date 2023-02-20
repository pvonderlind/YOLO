import torch
import torch.nn as nn

from yolo_v1 import ConvBlock, repeat_block


class YoloV2(nn.Module):

    def __init__(self):
        super().__init__()

        self.darknet_conv = nn.Sequential(
            ConvBlock(3, 32, kernel_size=(3, 3)),  # Out: 224x224
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # Out: 112x112
            ConvBlock(32, 64, kernel_size=(3, 3)),  # Out: 112x112
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # Out: 56x56
            ConvBlock(64, 128, kernel_size=(3, 3)),  # Out: 56x56
            ConvBlock(128, 64, kernel_size=(1, 1)),  # Out: 56x56
            ConvBlock(64, 128, kernel_size=(3, 3)),  # Out: 56x56
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # Out: 28x28
            ConvBlock(128, 256, kernel_size=(3, 3)),  # Out: 28x28
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # Out: 14x14
            *repeat_block([ConvBlock(256, 512, kernel_size=(3, 3)),
                          ConvBlock(512, 256, kernel_size=(1, 1))], 2),  # Out: 14x14
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # Out: 7x7
            *repeat_block([ConvBlock(512, 1024, kernel_size=(3, 3)),
                           ConvBlock(1024, 512)], 2),  # Out: 7x7
            ConvBlock(512, 1024, kernel_size=(3, 3)),  # Out: 7x7
        )

        self.darknet_final = nn.Sequential(
            ConvBlock(1024, 1000, kernel_size=(1, 1)),  # Out: 7x7
            nn.AvgPool2d(kernel_size=(7, 7)),  # Out: 1000
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.darknet_conv(x)
        x = self.darknet_final(x)
        return x
