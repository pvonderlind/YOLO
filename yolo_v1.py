import torch
from utils import intersection_over_union


def repeat_block(blocks: list[torch.nn.Module], repeats: int) -> list[torch.nn.Module]:
    modules = [block for _ in range(repeats) for block in blocks]
    return modules


class YoloV1(torch.nn.Module):

    def __init__(self, S: int = 7, B: int = 2, C: int = 20, final_layer_size: int = 496):
        super().__init__()

        self.C = C
        self.S = S
        self.B = B
        self.final_layer_size = final_layer_size

        self.block_1 = torch.nn.Sequential(
            ConvBlock(3, 64, kernel_size=(7, 7), stride=2, padding=3),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block_2 = torch.nn.Sequential(
            ConvBlock(64, 192, kernel_size=(3, 3), padding=1),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block_3 = torch.nn.Sequential(
            ConvBlock(192, 128, kernel_size=(1, 1), padding=0),
            ConvBlock(128, 256, kernel_size=(3, 3), padding=1),
            ConvBlock(256, 256, kernel_size=(1, 1), padding=0),
            ConvBlock(256, 512, kernel_size=(3, 3), padding=1),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        # BLOCK 4
        self.block_4 = torch.nn.Sequential(
            *repeat_block([ConvBlock(512, 256, kernel_size=(1, 1), padding=0),
                           ConvBlock(256, 512, kernel_size=(3, 3), padding=1)],
                          4),
            ConvBlock(512, 512, kernel_size=(1, 1), padding=0),
            ConvBlock(512, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        # BLOCK 5
        self.block_5 = torch.nn.Sequential(
            *repeat_block(
                [ConvBlock(1024, 512, kernel_size=(1, 1), padding=0),
                 ConvBlock(512, 1024, kernel_size=(3, 3), padding=1)],
                2),
            ConvBlock(1024, 1024, kernel_size=(3, 3), padding=1),
            ConvBlock(1024, 1024, kernel_size=(3, 3), stride=2, padding=1)
        )

        # BLOCK 6
        self.block_6 = torch.nn.Sequential(
            ConvBlock(1024, 1024, kernel_size=(3, 3), padding=1),
            ConvBlock(1024, 1024, kernel_size=(3, 3), padding=1)
        )

        # FINAL DENSE
        self.final_dense = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 7 * 1024, self.final_layer_size),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(0.1),
            # Last layer dims = (S, S, 30) where (C+B*5) = 30
            torch.nn.Linear(self.final_layer_size, self.S * self.S * (self.C + self.B * 5)),
            # Sigmoid activation to avoid zeros
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.final_dense(x)
        x = x.view(-1, self.S, self.S, self.C + self.B * 5)
        return x


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bnorm = torch.nn.BatchNorm2d(num_features=out_channels)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.lrelu(x)
        return x


class YoloLoss(torch.nn.Module):
    def __init__(self, S: int = 7, B: int = 2, C: int = 20, l_coord: float = 5, l_noobj: float = .5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.mse = torch.nn.MSELoss(reduction='sum')

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Size of the prediction tensor is explained as follows:
        Each grid cell of the SxS prediction tensor has 30 values -> SxSx30

        In the dataset, the targets has the following structure:
        First C (=20) values in the last dim are the class probabilities, the 21th the class score, and the last 
        4 for the coordinates (so 21st to 25th value)
        :param target: Tensor of shape (S, S, 25) -> 20 class probs + 1 score + 4 coordinates of actual box
        :param prediction: Tensor of shape (S, S, 30) ->20 class probs + 1 score + 4 box coords + 1 score + 4 box coords
        """

        # SETUP ----------------------------------------------------------------------

        # STEP 1: get the best box out of all (currently it's B=2 currently, will be generalized later)
        # for each cell.
        # returns an (S,S)
        iou_box1 = intersection_over_union(prediction[..., 21:25], target[..., 21:25])
        iou_box2 = intersection_over_union(prediction[..., 26:30], target[..., 21:25])
        # We add a new dimensions for box1/box2 at 0 on which we concat!
        ious = torch.cat([iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)], dim=0)
        # Both have shape (S,S)
        _, best_box_idxs = torch.max(ious, dim=0)
        # Next, we generate the identity function Iobj_i (is an object in cell i)
        target_box_exists = target[..., 20].unsqueeze(3)

        # BOX COORD LOSS -------------------------------------------------------------
        # LINE 1 of loss fn in paper, this implements the sum over the coords terms:
        box_pred = best_box_idxs * prediction[..., 26:30] + (1 - best_box_idxs) * prediction[..., 21:25]
        box_pred *= target_box_exists

        box_target = target_box_exists * target[..., 21:25]
        # LINE 2 of loss fn in paper, apply sqrt over the width and height of absolute values of width and height
        # In the paper they use
        # 1e-6 for numerical stability, use sign and abs to avoid problems with neg values
        # the beginning of the training.

        # (N, S, S, 4)
        box_pred[..., 2:4] = torch.sign(box_pred[..., 2:4]) * (box_pred[..., 2:4] + 1e-6).abs().sqrt()
        box_target[..., 2:4] = box_target[..., 2:4].sqrt()

        # flatten to (N * S * S, 4)
        box_loss = self.mse(box_pred.flatten(end_dim=-2), box_target.flatten(end_dim=-2))

        # OBJECT LOSS ----------------------------------------------------------------
        # Always use slices here to keep the dimensions intact!
        pred_obj_box = best_box_idxs * prediction[..., 25:26] + (1 - best_box_idxs) * prediction[..., 20:21]
        pred_obj_box *= target_box_exists
        target_obj_box = target_box_exists * target[..., 20:21]
        obj_loss = self.mse(pred_obj_box.flatten(), target_obj_box.flatten())

        # NO OBJECT LOSS -------------------------------------------------------------
        # Note: When there is no object, calculate loss for BOTH boxes!
        pred_no_obj_b1 = (1 - target_box_exists) * prediction[..., 20:21]
        pred_no_obj_b2 = (1 - target_box_exists) * prediction[..., 25:26]
        target_no_obj_flat = ((1 - target_box_exists) * target[..., 20:21]).flatten(start_dim=1)

        no_obj_loss = self.mse(pred_no_obj_b1.flatten(start_dim=1), target_no_obj_flat)
        no_obj_loss += self.mse(pred_no_obj_b2.flatten(start_dim=1), target_no_obj_flat)

        # CLASS LOSS -----------------------------------------------------------------
        pred_class_probs = target_box_exists * prediction[..., :20]
        target_class_probs = target_box_exists * target[..., :20]

        class_loss = self.mse(pred_class_probs.flatten(end_dim=-2), target_class_probs.flatten(end_dim=-2))

        return self.l_coord * box_loss + obj_loss + self.l_noobj * no_obj_loss + class_loss
