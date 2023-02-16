import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler

import utils
from yolo_dataset import PascalVocDataset, Compose
from yolo_v1 import YoloV1, YoloLoss
from utils import plot_gradient_updates, plot_predictions_vs_targets

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

BATCH_SIZE = 32
LR = 2e-5
SEED = 12345678
VAL_SPLIT = 0.0

S = 7
C = 20
B = 2

IMG_X_DIM = 448
IMG_Y_DIM = 448

LOG_INTERVAL = 100
EPOCHS = 20000

g = torch.Generator().manual_seed(SEED)
np.random.seed(SEED)


TRAIN_CSV = 'data/8examples.csv'
TEST_CSV = 'data/test.csv'

LOAD_MODEL = True
MODEL_PATH = 'yolov1_100.pt'
SAVE_MODEL = False


def get_dataloaders(transforms: Compose) -> tuple[DataLoader, DataLoader, DataLoader]:
    # BUILD TRAIN/VAL DATALOADERS --------------------------------------------------------------------------------------
    train_dataset = PascalVocDataset(S, B, C, TRAIN_CSV, transforms, device=device)
    train_size = len(train_dataset)
    indices = list(range(train_size))
    split = int(np.floor(VAL_SPLIT * train_size))
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, generator=g,
                                  sampler=train_sampler)
    val_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, generator=g, sampler=val_sampler)

    # BUILD TEST DATALOADER --------------------------------------------------------------------------------------------
    test_dataset = PascalVocDataset(S, B, C, TEST_CSV, transforms, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    return train_dataloader, val_dataloader, test_dataloader


def run_train_loop(train_loader, model, optimizer, loss_fn, debug=False):
    train_loader_tqdm = tqdm(train_loader, leave=True)
    gradient_updates = []
    lossi = []

    for batch_idx, (image, label) in enumerate(train_loader_tqdm):
        # Forward pass
        image, label = image.to(device), label.to(device)
        out = model(image)
        loss = loss_fn(out, label)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossi.append(loss.log10().item())
        train_loader_tqdm.set_postfix(loss=loss.item())

        # Debug Metrics
        if debug:
            with torch.no_grad():
                cur_lr = optimizer.param_groups[0]['lr']
                update = [(cur_lr * p.grad.std() / p.data.std()).log10().item() for p in model.parameters()]
                gradient_updates.append(update)
            if batch_idx % LOG_INTERVAL == 0:
                plot_gradient_updates(gradient_updates, model.parameters())
        return lossi, gradient_updates


def main():
    model = YoloV1(S, B, C).to(device)
    loss_fn = YoloLoss(S, B, C, l_noobj=0.5, l_coord=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    transform = Compose([transforms.Resize((IMG_Y_DIM, IMG_X_DIM))])

    train_loader, val_loader, test_loader = get_dataloaders(transform)
    # TODO: Remove debug setting after debugging!

    if LOAD_MODEL:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    try:
        losses = []
        grad_updates = []
        for _ in range(EPOCHS):

            # # TODO: remove this after testing
            # for x, y in train_loader:
            #     x = x.to(device)
            #     y = y.to(device)
            #     out = model(x)[0:1]
            #     y = y[0:1]
            #     plot_predictions_vs_targets(x[0], out, y)
            #     import sys
            #     sys.exit()

            lossi, grad_up_i = run_train_loop(train_loader, model, optimizer, loss_fn, debug=False)
            losses += lossi
            grad_up_i += grad_up_i
    except KeyboardInterrupt:
        # plt.plot(torch.tensor(losses).view(-1, 100).mean(1))
        plt.plot(losses)
        plt.title('log10 loss')
        plt.show()
    finally:
        if SAVE_MODEL:
            save_checkpoint(model, optimizer)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, MODEL_PATH)


if __name__ == "__main__":
    main()
