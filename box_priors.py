import os.path
import pandas as pd
from torchvision import ops
import torch

"""
Use this file to generate the bounding box priors for a given training
set (PascalVOC 2007, COCO) used for YOLO9000 (YOLOV2) and YOLOV3
"""

DATASET_FILE = 'data/train.csv'
LABEL_DIR = 'data/labels'

# In the paper of YOLOV2 the authors found that choosing K=5 yields the best results.
K = 5

def _get_boxes_tensor_from_dataset_labels(path_to_ds_file: str, label_dir: str) -> torch.Tensor():
    """
    :return: A tensor of bounding box labels read from the given files. The boxes have the format
    x_1,y_1,x_2,y_2, where x_2 and y_2 are generated by adding the box width and height to x_1/y_1.
    """
    image_labels = pd.read_csv(path_to_ds_file, sep=',', header=None)
    boxes = []
    for i in range(image_labels.shape[0]):
        label_filename = image_labels.iloc[i, 1]
        label_path = os.path.join(label_dir, label_filename)
        with open(label_path, 'r', encoding='utf-8') as file:
            for line in file.read().splitlines():
                # NOTE: The labels are in for c,x,y,w,h
                box = [float(val) for val in line.split(' ')]
                # --> For torch.ops.box_iou we want the format x1,y1,x2,y2
                # In the label file repesentation, x,y are the center of the box relative to the img!
                box.pop(0)  # Remove the class prediction
                x, y, w, h = box
                x_1 = x - w/2
                y_1 = y - h/2
                x_2 = x + w/2
                y_2 = y + h/2
                box_torch_format = [x_1, y_1, x_2, y_2]
                boxes.append(box_torch_format)
    return torch.tensor(boxes)


def _iou_kmeans(bboxes: torch.Tensor, k: int, stop_iter: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    :param bboxes: (N, 4) with box coordinates x_1, y_1, x_2, y_2 in relation to the image [0,1]
    :param k: Number k of clusters to find in the boxes.
    :param stop_iter: Defines after how many refinement iterations to stop.
    :return: Returns the cluster centroid boxes in relation to img in shape (k, 4) and the distances of
    all boxes to those cluster centroids.
    """
    n_boxes = bboxes.shape[0]
    distances = torch.empty((n_boxes, k))
    last_clusters = torch.empty((n_boxes,))

    # Randomly assign each box to a cluster centroid
    cluster_idxs = torch.randperm(n_boxes)[:k]
    clusters = bboxes[cluster_idxs].clone()

    iteration = 0
    while True:
        # (N, K)
        distances = 1 - ops.box_iou(bboxes, clusters)
        # Get cluster idx along the cluster axis of shape K
        # (N, 1) where the dim 1 is in range(K)
        nearest_clusters = torch.argmin(distances, dim=1)

        # Check if all cluster are in equilibrium
        # If they are, break if grace period is up
        if (last_clusters == nearest_clusters).all():
            iteration += 1
            if iteration == stop_iter:
                print(f"Halting kmeans, equilibrium found!")
                break
        else:
            iteration = 0

        # Generate new cluster centroids by taking the mean over all
        # the bboxes assigned to it!
        for cluster_idx in range(k):
            if cluster_idx in nearest_clusters:
                clusters[cluster_idx] = torch.mean(bboxes[nearest_clusters == cluster_idx], dim=0)
        last_clusters = nearest_clusters.clone()
    return clusters, distances


def main():
    bboxes = _get_boxes_tensor_from_dataset_labels(DATASET_FILE, LABEL_DIR)
    clusters, distances = _iou_kmeans(bboxes, K, stop_iter=1)
    print(clusters)


if __name__ == "__main__":
    main()