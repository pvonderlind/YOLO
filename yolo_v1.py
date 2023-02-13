import torch
import torch.functional as F


def repeat_block(blocks: list[torch.nn.Module], repeats: int) -> list[torch.nn.Module]:
    modules = [block for i in range(repeats) for block in blocks]
    return modules


class YoloV1(torch.nn.Module):

    def __init__(self, num_boxes: int = 2, n_classes: int = 20, split_size: int = 7, final_layer_size: int = 4096):
        super().__init__()

        self.C = n_classes
        self.S = split_size
        self.B = num_boxes
        self.final_layer_size = final_layer_size

        self.block_1 = torch.nn.Sequential(
            ConvBlock(3, 64, kernel_size=(7, 7), stride=2, padding=3),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.block_2 = torch.nn.Sequential(
            ConvBlock(64, 192, kernel_size=(3, 3), padding=1),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.block_3 = torch.nn.Sequential(
            ConvBlock(192, 128, kernel_size=(1, 1), padding=2),
            ConvBlock(128, 256, kernel_size=(3, 3)),
            ConvBlock(256, 256, kernel_size=(1, 1)),
            ConvBlock(256, 512, kernel_size=(3, 3)),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )

        # BLOCK 4
        self.block_4 = torch.nn.Sequential(
            *repeat_block([ConvBlock(512, 256, kernel_size=(1, 1)), ConvBlock(256, 512, kernel_size=(3, 3), padding=1)],
                          4),
            ConvBlock(512, 512, kernel_size=(1, 1)),
            ConvBlock(512, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # BLOCK 5
        self.block_5 = torch.nn.Sequential(
            *repeat_block(
                [ConvBlock(1024, 512, kernel_size=(1, 1)), ConvBlock(512, 1024, kernel_size=(3, 3), padding=1)], 2),
            ConvBlock(1024, 1024, kernel_size=(3, 3), padding=1),
            ConvBlock(1024, 1024, kernel_size=(3, 3), stride=2, padding=1)
        )

        # BLOCK 6
        self.block_6 = torch.nn.Sequential(
            ConvBlock(1024, 1024, kernel_size=(3, 3), padding='same'),
            ConvBlock(1024, 1024, kernel_size=(3, 3), padding='same')
        )

        # FINAL DENSE
        self.final_dense = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 7 * 1024, self.final_layer_size),
            torch.nn.Dropout(0.0),
            torch.nn.LeakyReLU(0.1),
            # Last layer dims = (S, S, 30) where (C+B*5) = 30
            torch.nn.Linear(self.final_layer_size, self.S * self.S * (self.C + self.B * 5))
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.final_dense(x)
        x = x.view(self.S, self.S, 30)
        return x


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bnorm = torch.nn.BatchNorm2d(num_features=out_channels)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.lrelu(x)
        return x
