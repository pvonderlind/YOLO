import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# TODO: Find right class assignments!
pascal_voc_classes = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]


def plot_predictions_vs_targets(image: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor):
    pred_boxes = get_box_list_from_tensor(predictions)
    pred_boxes = non_max_suppression(pred_boxes, 0.5)
    target_boxes = get_box_list_from_tensor(targets)
    target_boxes = non_max_suppression(target_boxes, 0.5)

    fig, ax = plt.subplots(1)
    image = image.to('cpu').long()
    _, width, height = image.shape
    ax.imshow(image.permute(1, 2, 0))

    _plot_box_predictions_on_image(width, height, ax, pred_boxes, target_boxes)

    plt.show()


def plot_predictions(image: torch.Tensor, predictions: torch.Tensor):
    pred_boxes = get_box_list_from_tensor(predictions)
    pred_boxes = non_max_suppression(pred_boxes, 0.5)

    fig, ax = plt.subplots(1)
    image = image.to('cpu').long()
    _, width, height = image.shape
    ax.imshow(image.permute(1, 2, 0))

    for box in pred_boxes:
        box_plt_rect = _get_box_rect(box, width, height, 'r')
        ax.add_patch(box_plt_rect)
    plt.show()


def _plot_box_predictions_on_image(width, height, ax, box_pred: list, box_target: list):
    for box_p, box_t in zip(box_pred, box_target):
        box_p_plot = _get_box_rect(box_p, width, height, 'r')
        box_t_plot = _get_box_rect(box_t, width, height, 'b')
        ax.add_patch(box_p_plot)
        ax.add_patch(box_t_plot)


def _get_box_rect(box: list, width, height, color: str):
    x, y, w, h = box[2:]
    l_up_x = x - w / 2
    l_up_y = y - h / 2

    img_x = l_up_x * width
    img_y = l_up_y * height
    img_w = w * width
    img_h = h * height

    plot_box = Rectangle(
        (img_x, img_y),
        img_w,
        img_h,
        linewidth=1,
        edgecolor=color,
        facecolor='none'
    )
    plt.annotate(pascal_voc_classes[int(box[1])], (img_x, img_y), color=color)
    return plot_box


def get_box_list_from_tensor(predictions: torch.Tensor) -> list:
    """
    :param predictions: shape (N, 7, 7, 6)
    :return: A list of boxes for each sample.
    """
    boxes = []
    cell_boxes = convert_box_predictions_to_cell_boxes(predictions)
    n_cells = predictions.shape[1] * predictions.shape[2]
    boxes_per_sample = cell_boxes.reshape(1, n_cells, -1)
    for j in range(n_cells):
        boxes.append([box.item() for box in boxes_per_sample[0, j, :]])
    return boxes


def convert_box_predictions_to_cell_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """
    :param boxes: (N, 7, 7, 30)
    :return: tensor of shape (N, 7, 7, 5) where 5 has the form (class proba, class pred, x, y, w, h)
    """
    boxes = boxes.to('cpu')
    # (N, 7, 7, 4)
    box1_coords = boxes[..., 21:25]
    box2_coords = boxes[..., 26:30]

    # (N, 7, 7, 1)
    box1_scores = boxes[..., 20:21]
    box2_scores = boxes[..., 25:26]

    # (2, N, 7, 7, 1)
    scores = torch.cat([box1_scores.unsqueeze(0), box2_scores.unsqueeze(0)], dim=0)
    # (N, 7, 7, 1) -> determines which of the two boxes has the higher score in which section
    _, best_box_section_idx = scores.max(dim=0)

    # (N, 7, 7, 4) -> Coords of the best box per section
    best_box_coords = box1_coords * (1 - best_box_section_idx) + best_box_section_idx * box2_coords
    best_box_img_coords = _convert_box_coords_to_img_coords_for(best_box_coords)

    # (N, 7, 7, 2)
    class_conf = _get_class_and_confidence(boxes)
    # (N, 7, 7, 6)
    return torch.cat([class_conf, best_box_img_coords], dim=-1)


def _get_class_and_confidence(boxes: torch.Tensor):
    # (N, 7, 7, 1)
    predicted_class = torch.round(boxes[..., 0:20].argmax(-1, keepdim=True))
    # (N, 7, 7, 1)
    confidence = torch.max(boxes[..., 20:21], boxes[..., 25:26])
    # (N, 7, 7, 2)
    return torch.cat([confidence, predicted_class], dim=-1)


def _convert_box_coords_to_img_coords_for(box_coord: torch.Tensor) -> torch.Tensor:
    """
    :param box_coord: (N, 7, 7, 4) tensor containing the coords of boxes for each cell (x,y,w,h)
    :return:
    """
    S = box_coord.shape[1]
    n = box_coord.shape[0]

    # CONVERT COORDINATES TO BE RELATIVE TO IMG

    # (N, 7, 7, 1) -> Contains the S values for each box, these will be broadcast to the best_boxes
    # reforming both the x,y,h,w values into larger S-related sizes (which need to be divided afterwards by 1/S)
    # Unsqueeze of last dim is necessary since broadcasting prepends the new dimension, instead of appending!!
    S_idxs = torch.arange(S).repeat(n, S, 1).unsqueeze(-1)
    # (N, 7, 7, 1)
    x_img = (box_coord[..., 0:1] + S_idxs) / S
    # (N, 7, 7, 1)
    y_img = (box_coord[..., 1:2] + S_idxs.permute(0, 2, 1, 3)) / S
    # (N, 7, 7, 2)
    h_w = box_coord[..., 2:4] / S

    return torch.cat([x_img, y_img, h_w], dim=-1)


def plot_gradient_updates(gradient_updates: list, parameters):
    plt.figure(figsize=(20, 4))
    legends = []
    for i, p in enumerate(parameters):  # exclude output layer
        plt.plot([gradient_updates[j][i] for j in range(len(gradient_updates))])
        legends.append(f'param {i}')
    plt.plot([0, len(gradient_updates)], [-3, -3], 'k')  # ratios should be at roughly or below 1e-3
    plt.legend(legends)
    plt.show()


def intersection_over_union(pred_box: torch.Tensor, target_box: torch.Tensor):
    """
    :param pred_box: shape (S, S, 4) -> (x,y,h,w) x,y is the center of the box with width w and height h
    :param target_box: shape (S, S, 4) -> (x,y,h,w)
    :return: IOU measure (intersection area divided by union area) -> closer to 1 is better
    """
    pred_x_left, pred_y_left, pred_x_right, pred_y_right = _get_left_right_coords(pred_box)
    target_x_left, target_y_left, target_x_right, target_y_right = _get_left_right_coords(target_box)

    x_left = torch.max(pred_x_left, target_x_left)
    x_right = torch.min(pred_x_right, target_x_right)

    y_left = torch.max(pred_y_left, target_y_left)
    y_right = torch.min(pred_y_right, target_y_right)

    intersection_areas = (x_right - x_left).clamp(0) * (y_right - y_left).clamp(0)

    pred_area = torch.abs((pred_x_right - pred_x_left) * (pred_y_right - pred_y_left))
    target_area = torch.abs((target_x_right - target_x_left) * (target_y_right - target_y_left))

    return intersection_areas / (pred_area + target_area - intersection_areas + 1e-6)


def _get_left_right_coords(box: torch.Tensor) -> tuple:
    left_x = box[..., 0:1] - box[..., 2:3] / 2
    left_y = box[..., 1:2] - box[..., 3:4] / 2
    right_x = box[..., 0:1] + box[..., 2:3] / 2
    right_y = box[..., 1:2] + box[..., 3:4] / 2
    return left_x, left_y, right_x, right_y


def non_max_suppression(boxes: list, iou_threshold: float):
    selected_boxes = []

    for box_i in boxes:
        discard = False
        box_i_t = torch.tensor(box_i[2:])
        for box_j in boxes:
            box_j_t = torch.tensor(box_j[2:])
            iou = intersection_over_union(box_i_t, box_j_t)
            if iou < iou_threshold:
                if box_j[0] > box_i[0]:
                    discard = True
        if not discard:
            selected_boxes.append(box_i)
    return selected_boxes
