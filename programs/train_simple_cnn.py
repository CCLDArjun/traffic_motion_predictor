import sys

sys.path.append("../")

import time
import pickle
import os
from unittest import mock
import math

import torch
from torch import _dynamo as torchdynamo
from torch.utils.tensorboard import SummaryWriter
from models.simple_cnn import SimpleCNN, loss_function

from dataset import SimpleCNNDataset, DictDataset, ChunkReader
from dataset.simple_cnn_dataset import denormalize_xy
from utils import (
    random_id,
    get_time_str,
    nsight_profiler,
    profiler,
    startNsight,
    stopNsight,
)
from utils.metrics import minADE, minFDE

torch.autograd.set_detect_anomaly(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch._dynamo.config.suppress_errors = True
# torch.cuda.set_sync_debug_mode(1)

cpu = False
PROFILE = False
MEMORY_PROFILE = False
CHECK_NAN = False

NSIGHT_PROFILE = False
WARMUP_EPOCHS = 1

NUM_MODES = 25
PREDS_PER_MODE = 12

EPOCHS = 92 + WARMUP_EPOCHS
BATCH_SIZE = 28
print("BATCH SIZE:", BATCH_SIZE)
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
REPORT_INTERVAL = 15  # batches
PRINT_INTERVAL = 15  # batches

PICKLE_DATASET = False
PICKLE_DATASET_PATH = "simple_cnn_dataset.pickle"

CHUNK_DATASET = True
CHUNK_DATASET_PATH = "./chunks/"

START_AT_CHECKPOINT = False
CHECKPOINT = "./runs/simple_cnn_rVW9hZ6wZcpHqLIen3Tio1r8haPBGTmtcUitSfVfJ_e39.pt"
WRITE_TENSORBOARD = True

if CHUNK_DATASET:
    d = ChunkReader(CHUNK_DATASET_PATH)
    print("LOADED CHUNK DATASET")
elif not os.path.isfile(PICKLE_DATASET_PATH) and PICKLE_DATASET:
    d = SimpleCNNDataset("../data/sets/v1.0-trainval", size="full", cache_size=0)

    arr = []
    start = time.time()
    for i in range(len(d)):
        if i % 500 == 499:
            print("Loaded", i, "in ", time.time() - start, "seconds")
        arr.append(d[i])

    torch.save(DictDataset(arr), PICKLE_DATASET_PATH)
    print("PICKLED DATASET")

elif PICKLE_DATASET:
    d = torch.load(PICKLE_DATASET_PATH)
    print("LOADED PICKLED DATASET")
else:
    d = SimpleCNNDataset("../data/sets/v1.0-mini", size="mini")

if START_AT_CHECKPOINT:
    run_id = CHECKPOINT.split("_")[-2][1:]
else:
    run_id = random_id(40)

print("Run ID:", run_id)

model = SimpleCNN(
    num_modes=NUM_MODES,
    predictions_per_mode=PREDS_PER_MODE,
    state_vector_size=sample[1].shape[0],
)

if START_AT_CHECKPOINT:
    model.load_state_dict(torch.load(CHECKPOINT)["model_state_dict"])

device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
is_cuda = device.type == "cuda"
model.to(device)

# compile model.forward to make it faster
# model_forward = torch.jit.trace(model.forward, example_inputs=(sample[0].to(device), sample[1].to(device)))

train, val = torch.utils.data.random_split(
    d,
    [int(len(d) * 0.8), len(d) - int(len(d) * 0.8)],
    generator=torch.Generator().manual_seed(42),
)
training_loader = torch.utils.data.DataLoader(
    train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=is_cuda,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=5,
)
validation_loader = torch.utils.data.DataLoader(
    val,
    batch_size=3 * BATCH_SIZE,
    shuffle=True,
    pin_memory=is_cuda,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=5,
)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

if START_AT_CHECKPOINT:
    optimizer.load_state_dict(torch.load(CHECKPOINT)["optimizer_state_dict"])

if WRITE_TENSORBOARD:
    tb_writer = SummaryWriter(log_dir="runs/simple_cnn_r" + run_id)
    tb_writer.add_text("run_id", run_id)
else:
    tb_writer = mock.Mock()
    tb_writer.add_scalar = lambda *args, **kwargs: None

if PROFILE:
    model.forward = profiler(model.forward, "forward")
    loss_function = profiler(loss_function, "loss")
    optimizer.step = profiler(optimizer.step, "optstep")
if NSIGHT_PROFILE:
    model.forward = nsight_profiler(model.forward, "forward")
    loss_function = nsight_profiler(loss_function, "loss")
    optimizer.step = nsight_profiler(optimizer.step, "optstep")


def calculate_metrics(predictions, probabilities, gt):
    # denormalize predictions and ground truth
    predictions = denormalize_xy(predictions)
    gt = denormalize_xy(gt)

    minADE_ = minADE(predictions, probabilities, gt, k=NUM_MODES).sum().item()
    minFDE_ = minFDE(predictions, probabilities, gt, k=NUM_MODES).sum().item()

    return {
        "minADE": minADE_,
        "minFDE": minFDE_,
    }


def _preprocess(data):
    for i, d_ in enumerate(data):
        data[i] = d_.cuda(non_blocking=True)
    data[0] = data[0].squeeze(1)
    return data


def train_one_epoch(
    epoch_index, model, optimizer, dataloader, val_dataloader, device, tb_writer
):
    model.train()
    running_loss = 0
    running_l1_loss = 0
    running_minADE = 0
    running_minFDE = 0
    last_loss = 0

    for i, data in enumerate(dataloader):
        # SimpleCNNDataset returns shape (1, C, H, W), dataloader returns (B, 1, C, H, W)
        # squeeze to remove the 1 dimension

        data = _preprocess(data)

        if CHECK_NAN:
            for _, d_ in enumerate(data):
                if d_.isnan().any():
                    print("NAN FOUND")
                    breakpoint()

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            y_pred = model(data[0], data[1])
            loss, l1_loss = loss_function(*y_pred, data[2])

        if PROFILE:
            profiler(loss.backward, "backward")()
        if NSIGHT_PROFILE:
            nsight_profiler(loss.backward, "backward")()
        else:
            loss.backward()

        running_loss += loss.item()
        running_l1_loss += l1_loss

        metrics = calculate_metrics(*y_pred, data[2])
        running_minADE += metrics["minADE"]
        running_minFDE += metrics["minFDE"]

        optimizer.step()

        # this was originally done with BATCH_SIZE = 16 so the tb_x is calculated wrong for a different BATCH_SIZE
        # so we should calculate it as if BATCH_SIZE = 16
        tb_x = math.ceil(epoch_index * len(dataloader) * BATCH_SIZE / 16) + i + 1

        if i % REPORT_INTERVAL == REPORT_INTERVAL - 1:
            running_loss /= REPORT_INTERVAL - 1  # loss per batch
            running_l1_loss /= REPORT_INTERVAL - 1
            running_minADE /= REPORT_INTERVAL - 1
            running_minFDE /= REPORT_INTERVAL - 1

            tb_writer.add_scalar("Loss/train", running_loss, tb_x)
            tb_writer.add_scalar("Loss/l1_loss", running_l1_loss, tb_x)
            tb_writer.add_scalar("Metrics/minADE", running_minADE, tb_x)
            tb_writer.add_scalar("Metrics/minFDE", running_minFDE, tb_x)

            running_loss = 0.0
            running_l1_loss = 0.0
            running_minADE = 0.0
            running_minFDE = 0.0

        if i == len(dataloader) - 1:  # last batch
            start = time.time()
            losses, l1_losses, minADEs, minFDEs = (
                torch.empty(len(val_dataloader)),
                torch.empty(len(val_dataloader)),
                torch.empty(len(val_dataloader)),
                torch.empty(len(val_dataloader)),
            )
            for j, val_data in enumerate(val_dataloader):
                _preprocess(val_data)

                model.eval()
                with torch.no_grad():
                    y_pred = model(val_data[0], val_data[1])
                loss, l1_loss = loss_function(*y_pred, val_data[2])
                model.train()

                metrics = calculate_metrics(*y_pred, val_data[2])

                losses[j] = loss.item()
                l1_losses[j] = l1_loss
                minADEs[j] = metrics["minADE"]
                minFDEs[j] = metrics["minFDE"]

            tb_writer.add_scalar("Loss/val", torch.mean(losses), tb_x)
            tb_writer.add_scalar("Loss/l1_loss_val", torch.mean(l1_losses), tb_x)
            tb_writer.add_scalar("Metrics/minADE_val", torch.mean(minADEs), tb_x)
            tb_writer.add_scalar("Metrics/minFDE_val", torch.mean(minFDEs), tb_x)


if __name__ == "__main__":
    if MEMORY_PROFILE:
        torch.cuda.memory._record_memory_history()

    # if checkpoint, start at the next epoch
    if START_AT_CHECKPOINT:
        start = CHECKPOINT.split("_")[-1].split(".")[0][1:]
        start = int(start) + 1
        print("continuing from epoch", start)
    else:
        start = 1

    for x in range(start, EPOCHS + 1):
        if NSIGHT_PROFILE and x == WARMUP_EPOCHS + 1:
            startNsight()
        start = time.time()
        # profiler(train_one_epoch, "big_profile")(x, model, optimizer, training_loader, device, tb_writer)
        train_one_epoch(
            x, model, optimizer, training_loader, validation_loader, device, tb_writer
        )
        end = time.time()
        print(f"Epoch {x} took {end - start} seconds")

        if WRITE_TENSORBOARD:
            torch.save(
                {
                    "epoch": x,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "NUM_MODES": NUM_MODES,
                    "PREDS_PER_MODE": PREDS_PER_MODE,
                    "BATCH_SIZE": BATCH_SIZE,
                    "INIT_LEARNING_RATE": LEARNING_RATE,
                    "INIT_MOMENTUM": MOMENTUM,
                },
                f"runs/simple_cnn_r{run_id}_e{x}.pt",
            )
    if MEMORY_PROFILE:
        torch.cuda.memory._dump_snapshot("memory_profiler_dump.pickle")
    if NSIGHT_PROFILE:
        stopNsight()
