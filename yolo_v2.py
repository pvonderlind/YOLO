import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_v1 import ConvBlock, repeat_block
from utils import intersection_over_union


# TODO: Add B=5 constants for the anchor box priors (p_w, p_h)
# TODO: Add logistic activation function to constrain network predictions to the range [0,1]
class YoloV2(nn.Module):

    def __init__(self, C: int = 20, B: int = 5, mode: str = 'classification'):
        super().__init__()
        self.mode = mode

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

        if self.mode == 'classification':
            self.darknet_final = nn.Sequential(
                ConvBlock(1024, 1000, kernel_size=(1, 1), padding='same'),  # Out: 7x7
                nn.AvgPool2d(kernel_size=(7, 7)),  # Out: 1000
            )
            self.softmax = nn.Softmax(dim=-1)
        else:
            # TODO: Add pass-through from last 512 feature conv layer (e.g. recurrent connection)
            n_detection_filters = (C + B) * 5
            self.darknet_final = nn.Sequential(
                *repeat_block([ConvBlock(1024, 1024, kernel_size=(3, 3), padding='same')], 3),
                # Use normal Conv2D since we don't want BN and ReLu for the last layer
                nn.Conv2d(1024, n_detection_filters, kernel_size=(1, 1)),
                nn.AvgPool2d(kernel_size=(1, 1))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (N, img_X, img_Y)
        :return: For detection a (N, dim, dim, 125) tensor, where the last dimension consist of
        5 boundings boxes with the shape 20 * class probs + class pred + t_x + t_y + t_w + t_h
        """
        x = self.darknet_conv(x)
        x = self.darknet_final(x)
        if self.mode == 'classification':
            x = x.view(-1, 1000)
            x = self.softmax(x)
        elif self.mode == 'detection':
            x = x.permute(0, 2, 3, 1)
        return x


def anchor_predictions_to_bounding_box(anchor_predictions: torch.Tensor, bbox_priors: torch.Tensor) -> torch.Tensor:
    """
    Convert anchor boxes predicted by YOLOV2 to bounding boxes used by Pascal VOC, YOlOV1 and
    COCO.

    :param bbox_priors: A (k, 2) tensor of k bounding box cluster centroids calculated from the training data
    that are used as priors for the 5 boxes of the anchor prediction.
    :param anchor_predictions: (N, S, S, (C + 5) * B)
    :return: A tensor containing the anchor box predictions converted to bounding box predictions (N, S, S, (B) * 5)
    """
    S = anchor_predictions.shape[1]
    n = anchor_predictions.shape[0]

    # Input shape: 0-19 = class probs, 20 = class pred, 21 = t_x, 22 = t_y, 23 = t_w, 24 = t_h

    # Steps 1)
    # Convert t_x, t_y to b_x, b_y by putting the predictions through a sigmoid and then adding the cell coords c_x, c_y
    # (N, S, S, 5)
    sig_tx = torch.sigmoid(anchor_predictions[..., 21::25])
    sig_ty = torch.sigmoid(anchor_predictions[..., 22::25])

    # (N, S, S, 1)
    cell_coords = torch.arange(S).repeat(n, S, 1).unsqueeze(-1)
    # Add cell coordinates to x, y respectively, Results are (N, S, S, 5)
    b_x = (sig_tx + cell_coords) / S
    # Note: swap dimensions of the cells here to account for the direction of y!
    b_y = (sig_ty + cell_coords.permute(0, 2, 1, 3)) / S

    # Step 2)
    # Convert anchor box width and height to bounding box width and height by using the pre-defined box priors
    # for B=5 boxes and multiplying them with exp(t_w) and exp(t_h) respectively.
    b_w = anchor_predictions[..., 23::25].exp() * bbox_priors[..., 0]
    b_h = anchor_predictions[..., 24::25].exp() * bbox_priors[..., 1]

    return torch.cat([b_x, b_y, b_w, b_h], dim=-1)


class YoloV2Loss(nn.Module):

    def __init__(self, bbox_priors: torch.Tensor, l_coords: float = 5, l_noobj: float = .5):
        super().__init__()
        self.l_coords = l_coords
        self.l_noobj = l_noobj
        self.bbox_priors = bbox_priors
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        YOLOV2 loss is adapted from YOLOV1 loss. A good explanation can be found in this
        StackOverflow post: https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation

        The loss function is pretty much the same, except that we need to do it for 5 boxes and first convert
        the anchor boxes to bounding box coordinates.
        :param prediction: Tensor of shape (N, S, S, 125) with anchor box predictions using 5 bbox priors
        :param target: Tensor of shape (N, S, S, 25) -> 20 class probs + 1 score + 4 coordinates of actual box
        """

        # SETUP --------------------------------------------------------------------------------------------------------

        S = prediction.shape[1]
        # (N, S, S, 20) 5 = x, y, w, h of the bounding box (x,y is the center!)
        b_coords = anchor_predictions_to_bounding_box(prediction, self.bbox_priors)
        # (N, S, S, 5, 4)
        b_coords = b_coords.view(-1, S, S, 5, 4)

        # NOTE: THIS IS UGLY, BUT VECTORIZED SO IT'S FASTER!!
        # TODO: Find a more beautiful representation of this that is vectorized ...
        # Shapes are (N, 7, 7)
        iou_box_1 = intersection_over_union(b_coords[..., 0, :], target[..., 21:25]).unsqueeze(0)
        iou_box_2 = intersection_over_union(b_coords[..., 1, :], target[..., 21:25]).unsqueeze(0)
        iou_box_3 = intersection_over_union(b_coords[..., 2, :], target[..., 21:25]).unsqueeze(0)
        iou_box_4 = intersection_over_union(b_coords[..., 3, :], target[..., 21:25]).unsqueeze(0)
        iou_box_5 = intersection_over_union(b_coords[..., 4, :], target[..., 21:25]).unsqueeze(0)

        # Get best box idx for each cell
        ious = torch.cat([iou_box_1, iou_box_2, iou_box_3, iou_box_4, iou_box_5], dim=0)
        # (S,S,1)
        _, best_box_idx = torch.max(ious, dim=0)
        # Next generate the identify function of objects in a cell
        # (N, S, S, 1)
        target_box_exists = target[..., 20].unsqueeze(3)

        # BOX COORD LOSS -----------------------------------------------------------------------------------------------
        # Get right box coords per iou maxes using some dimension magic:
        # A short explanation of this is: We first expand the best box indices
        # to match the shape of the coordinate tensor. Then, we gather along the third
        # axis of the b_coords (dim 3 has size 5, one dim per prior box) and use the expanded
        # matrix as index. Since this creates 5 equal elements for each result in dim 3, we take
        # only the 0th one and all coordinates from the last dimension.
        # TODO: Find a rework for this that is a bit more readable.
        best_box_idx_exp = best_box_idx.unsqueeze(-1).expand(-1, -1, -1, 5, 4)
        box_pred = torch.gather(b_coords, 3, best_box_idx_exp)[..., 0, :]
        box_pred *= target_box_exists

        box_target = target_box_exists * target[..., 21:25]

        # (N, S, S, 4)
        box_pred[..., 2:4] = torch.sign(box_pred[..., 2:4]) * (box_pred[..., 2:4] + 1e-6).abs().sqrt()
        box_target[..., 2:4] = box_target[..., 2:4].sqrt()

        # flatten to (N * S * S, 4)
        box_loss = self.mse(box_pred.flatten(end_dim=-2), box_target.flatten(end_dim=-2))

        # OBJECT LOSS
        all_object_pred = prediction[..., 20::25]
        best_object_pred = torch.gather(all_object_pred, 3, best_box_idx)
        best_object_pred *= target_box_exists
        target_object = target_box_exists * target[..., 20:21]
        obj_loss = self.mse(best_object_pred.flatten(), target_object.flatten())
        print('test')


if __name__ == "__main__":
    test = YoloV2(mode='detection')
    bbox_priors = torch.tensor([[0, 0, 1, 1]]).expand(5, -1)
    loss = YoloV2Loss(bbox_priors)
    test_input = torch.randint(0, 255, (1, 3, 416, 416), dtype=torch.float)
    out = test(test_input)
    test_loss = loss(out, torch.randint(0, 1, (1, 13, 13, 25)))
    print(out.shape)
