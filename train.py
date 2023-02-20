import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import wandb

from yolo_dataset import PascalVocDataset, Compose
from yolo_v1 import YoloV1, YoloLoss
from utils import plot_predictions_vs_targets

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

BATCH_SIZE = 16
LR = 2e-5
WEIGHT_DECAY = 0.0005
SEED = 12345678
VAL_SPLIT = 0.1

S = 7
C = 20
B = 2

IMG_X_DIM = 448
IMG_Y_DIM = 448

EVAL_INTERVAL = 20
EPOCHS = 200

g = torch.Generator().manual_seed(SEED)
np.random.seed(SEED)

TRAIN_CSV = 'data/train.csv'
TEST_CSV = 'data/test.csv'

LOAD_MODEL = True
MODEL_PATH = 'yolo_v1_full.pt'
SAVE_MODEL = False

LOG_RUN_TO_WANDB = True
WANDB_PROJECT = 'yolo_v1'

if LOG_RUN_TO_WANDB:
    wandb.init(
        project=WANDB_PROJECT,
        config={
            "model_file": MODEL_PATH,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "architecture": "YOLO_V1",
            "dataset": f"Pascal_VOC_{TRAIN_CSV}",
            "epochs": EPOCHS
        }
    )


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


def run_train_loop(train_loader: DataLoader, model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
                   debug: bool = False):
    train_loader_tqdm = tqdm(train_loader, leave=True)

    for batch_idx, (image, label) in enumerate(train_loader_tqdm):
        # Forward pass
        image, label = image.to(device), label.to(device)
        out = model(image)
        loss = loss_fn(out, label)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_i = loss.item()
        train_loader_tqdm.set_postfix(loss=loss_i)

        if LOG_RUN_TO_WANDB:
            wandb.log({'train_loss': loss_i})

        # Debug Metrics
        if debug:
            with torch.no_grad():
                cur_lr = optimizer.param_groups[0]['lr']
                wandb.log({'learning_rate': cur_lr})
                grad_updates = {name: (cur_lr * p.grad.std() / p.data.std()).log10().item() for name, p in
                                model.named_parameters()}
                wandb.log(grad_updates)


def plot_samples(loader: torch.utils.data.DataLoader, model: torch.nn.Module, n: int = 3):
    for x, y in loader:
        for idx in range(max(x.shape[0], n)):
            x = x.to(device)
            y = y.to(device)
            out_idx = model(x)[idx:idx + 1]
            y_idx = y[idx:idx + 1]
            plot_predictions_vs_targets(x[idx], out_idx, y_idx)


def main():
    model = YoloV1(S, B, C).to(device)
    loss_fn = YoloLoss(S, B, C, l_noobj=0.5, l_coord=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    transform = Compose([transforms.Resize((IMG_Y_DIM, IMG_X_DIM))])

    train_loader, val_loader, test_loader = get_dataloaders(transform)

    if LOAD_MODEL:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    try:
        for _ in range(EPOCHS):
            run_train_loop(train_loader, model, optimizer, loss_fn, debug=True)

            val_loss_avg = get_val_losses_avg_log10(model, loss_fn, val_loader)
            if LOG_RUN_TO_WANDB:
                wandb.log({'val_loss_avg': val_loss_avg})

    except KeyboardInterrupt:
        pass
    finally:
        if SAVE_MODEL:
            save_checkpoint(model, optimizer)


@torch.no_grad()
def get_val_losses_avg_log10(model: torch.nn.Module, loss_fn: torch.nn.Module, val_loader: DataLoader) -> float:
    model.eval()

    val_losses = []
    for batch_idx, (image, label) in enumerate(val_loader):
        image, label = image.to(device), label.to(device)
        out = model(image)
        loss = loss_fn(out, label)
        val_losses.append(loss.item())

    model.train()
    return sum(val_losses) / len(val_losses)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, MODEL_PATH)


if __name__ == "__main__":
    main()
