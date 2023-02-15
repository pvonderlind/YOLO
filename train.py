import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from yolo_dataset import PascalVocDataset, Compose
from yolo_v1 import YoloV1, YoloLoss

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

BATCH_SIZE = 1
LR = 0.0001
SEED = 12345678
VAL_SPLIT = 0.1

S = 7
C = 20
B = 2

IMG_X_DIM = 448
IMG_Y_DIM = 448

LOG_INTERVAL = 100

g = torch.Generator().manual_seed(SEED)
np.random.seed(SEED)


def get_dataloaders(transforms: Compose) -> tuple[DataLoader, DataLoader, DataLoader]:
    # BUILD TRAIN/VAL DATALOADERS --------------------------------------------------------------------------------------
    train_dataset = PascalVocDataset(S, B, C, 'data/train.csv', transforms, device=device)
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
    test_dataset = PascalVocDataset(S, B, C, 'data/test.csv', transforms, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    return train_dataloader, val_dataloader, test_dataloader


def run_train_loop(train_loader, model, optimizer, loss_fn, debug=False):
    train_loader_tqdm = tqdm(train_loader, leave=True)
    gradient_updates = []

    for batch_idx, (image, label) in enumerate(train_loader_tqdm):
        # Forward + Backward pass through model
        image, label = image.to(device), label.to(device)
        out = model(image)
        loss = loss_fn(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loader_tqdm.set_postfix(loss=loss.item())

        # Debug Metrics
        if debug:
            with torch.no_grad():
                cur_lr = optimizer.param_groups[0]['lr']
                update = [(cur_lr * p.grad.std() / p.data.std()).log10().item() for p in model.parameters()]
                gradient_updates.append(update)
            if batch_idx % LOG_INTERVAL == 0 and batch_idx != 0:
                plot_gradient_updates(gradient_updates, model.parameters())


def main():
    model = YoloV1(S, B, C).to(device)
    loss_fn = YoloLoss(S, B, C).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    transform = Compose([transforms.Resize((IMG_Y_DIM, IMG_X_DIM))])

    train_loader, val_loader, test_loader = get_dataloaders(transform)
    # TODO: Remove debug setting after debugging!
    run_train_loop(train_loader, model, optimizer, loss_fn, debug=True)


def plot_gradient_updates(gradient_updates: list, parameters):
    plt.figure(figsize=(20, 4))
    legends = []
    for i, p in enumerate(parameters):  # exclude output layer
        plt.plot([gradient_updates[j][i] for j in range(len(gradient_updates))])
        legends.append(f'param {i}')
    plt.plot([0, len(gradient_updates)], [-3, -3], 'k')  # ratios should be at roughly or below 1e-3
    plt.legend(legends)
    plt.show()


if __name__ == "__main__":
    main()
