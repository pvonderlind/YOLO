from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import torch
import os


class Compose(object):
    """
    Custom transform, since we don't want to transform the label tensor!
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, bounding_boxes: torch.Tensor) -> tuple:
        for transform in self.transforms:
            image = transform(image)
        return image, bounding_boxes


class PascalVocDataset(Dataset):

    def __init__(self, S: int, B: int, C: int, dataset_files_csv: str, transforms: Compose, device: str = 'cuda'):
        self.S = S
        self.B = B
        self.C = C

        self.device = device
        self.transforms = transforms

        self.image_label_paths = pd.read_csv(dataset_files_csv, sep=',', header=None)
        self.image_label_paths.columns = ['image', 'label']
        self.label_dir = 'data/labels/'
        self.img_dir = 'data/images/'
        pass

    def __len__(self):
        return self.image_label_paths.shape[0]

    def __getitem__(self, idx):
        """
        File content format is as follows:
        Multiple lines per label, each corresponds to a box target in the image.
        Each line contains:
        class_label (int), x (float, rel to img), y (float, rel to img), w (float, rel to img), h (float, rel to img)
        """
        label_path = os.path.join(self.label_dir, self.image_label_paths.loc[idx, 'label'])
        img_path = os.path.join(self.img_dir, self.image_label_paths.loc[idx, 'image'])

        boxes = []
        label = torch.zeros((self.S, self.S, self.C + 5 * self.B), device=self.device)
        image = read_image(img_path)

        with open(label_path, 'r', encoding='utf-8') as file:
            for line in file.read().splitlines():
                box = [float(val) for val in line.split(' ')]
                box[0] = int(box[0])
                boxes.append(box)

        for box in boxes:
            # Note: x,y are relative to the images whole size now -> [0,1]
            class_label, x, y, w, h = box
            S_row_idx = int(self.S * x)
            S_col_idx = int(self.S * y)
            # Convert coordinates of box relative to the cell
            x_rel_S = self.S * x - S_row_idx
            y_rel_S = self.S * y - S_col_idx
            # Same thing for width and height
            w_rel_S = w * self.S
            h_rel_S = h * self.S

            # Only set new box if there is none already!
            if label[S_row_idx, S_col_idx, 20] == 0.0:
                label[S_row_idx, S_col_idx, 20] = 1.0
                # Box position and class score
                label[S_row_idx, S_col_idx, 21:25] = torch.tensor([x_rel_S, y_rel_S, w_rel_S, h_rel_S],
                                                                  device=self.device)
                # Onehot encoding of class score = probabilities of target
                label[S_row_idx, S_col_idx, class_label] = 1.0

        image, label = self.transforms(image, label)
        return image.float(), label
