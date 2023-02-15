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
                class_score, x, y, w, h = box
                x_S = int(self.S * x)
                y_S = int(self.S * y)
                # Convert coordinates of box relative to the cell
                x_rel_S = self.S * x - x_S
                y_rel_S = self.S * y - y_S
                # Same thing for width and height
                w_rel_S = w * self.S
                h_rel_S = h * self.S

                # Only set new box if there is none already!
                if label[x_S, y_S, 20] == 0.0:
                    label[x_S, y_S, 20] = 1.0
                    # Box position and class score
                    label[x_S, y_S, -5:] = torch.tensor([class_score, x_rel_S, y_rel_S, w_rel_S, h_rel_S],
                                                        device=self.device)
                    # Onehot encoding of class score = probabilities of target
                    label[x_S, y_S, class_score] = 1.0

        image, label = self.transforms(image, label)
        return image.float(), label
