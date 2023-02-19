import torch
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import tkinter
from tkinter import filedialog

from yolo_v1 import YoloV1
from utils import plot_predictions
from yolo_dataset import Compose

MODEL_PATH = 'yolo_v1_full.pt'
IMG_X_DIM = 448
IMG_Y_DIM = 448

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

tkinter.Tk().withdraw()


def main():
    print(f'Loading model {MODEL_PATH} ...')
    model = YoloV1()
    model = model.to(DEVICE)
    transform = Compose([transforms.Resize((IMG_X_DIM, IMG_Y_DIM))])

    model.eval()
    file_names = filedialog.askopenfilenames()
    print(f"Annotating following files: {file_names}")

    images = []
    for f in file_names:
        img, _ = transform(read_image(f, mode=ImageReadMode.RGB), torch.tensor([]))
        images.append(img)

    img_batch = torch.stack(images, dim=0).float().to(DEVICE)
    out = model(img_batch)
    for i in range(out.shape[0]):
        plot_predictions(img_batch[i], out[i:i+1])


if __name__ == "__main__":
    main()
