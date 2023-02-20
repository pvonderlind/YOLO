import torch
import torch.nn as nn

from yolo_v1 import ConvBlock, repeat_block


class YoloV2(nn.Module):

    def __init__(self):
        super().__init__()

        self.darknet_conv = nn.Sequential(
            ConvBlock(3, 32, kernel_size=(3, 3), padding='same'),  # Out: 224x224
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # Out: 112x112
            ConvBlock(32, 64, kernel_size=(3, 3), padding='same'),  # Out: 112x112
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # Out: 56x56
            ConvBlock(64, 128, kernel_size=(3, 3), padding='same'),  # Out: 56x56
            ConvBlock(128, 64, kernel_size=(1, 1), padding='same'),  # Out: 56x56
            ConvBlock(64, 128, kernel_size=(3, 3), padding='same'),  # Out: 56x56
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # Out: 28x28
            ConvBlock(128, 256, kernel_size=(3, 3), padding='same'),  # Out: 28x28
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # Out: 14x14
            *repeat_block([ConvBlock(256, 512, kernel_size=(3, 3), padding='same'),
                           ConvBlock(512, 256, kernel_size=(1, 1), padding='same')], 2),  # Out: 14x14
            ConvBlock(256, 512, kernel_size=(3, 3), padding='same'),  # Out: 14x14
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # Out: 7x7
            *repeat_block([ConvBlock(512, 1024, kernel_size=(3, 3), padding='same'),
                           ConvBlock(1024, 512, kernel_size=(1, 1), padding='same')], 2),  # Out: 7x7
            ConvBlock(512, 1024, kernel_size=(3, 3), padding='same'),  # Out: 7x7
        )

        self.darknet_final = nn.Sequential(
            ConvBlock(1024, 1000, kernel_size=(1, 1), padding='same'),  # Out: 7x7
            nn.AvgPool2d(kernel_size=(7, 7)),  # Out: 1000
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.darknet_conv(x)
        x = self.darknet_final(x)
        x = x.view(-1, 1000)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    test = YoloV2()
    test_input = torch.randn((20, 3, 224, 224))
    out = test(test_input)
    print(out.shape)
    print(out[0].sum())
