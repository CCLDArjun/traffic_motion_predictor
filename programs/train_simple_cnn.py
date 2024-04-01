import sys
sys.path.append("../")

import time
import pickle
import os

import torch
from torch import _dynamo as torchdynamo
from torch.utils.tensorboard import SummaryWriter
from models.simple_cnn import SimpleCNN, loss_function

from dataset import SimpleCNNDataset, DictDataset, ChunkReader
from dataset.simple_cnn_dataset import denormalize_xy
from utils import random_id, get_time_str, nsight_profiler, profiler, startNsight, stopNsight
from utils.metrics import minADE, minFDE

torch.autograd.set_detect_anomaly(False)
#torch.cuda.set_sync_debug_mode(1)

cpu = False
PROFILE = False
MEMORY_PROFILE = False
CHECK_NAN = False

NSIGHT_PROFILE = False
WARMUP_EPOCHS = 1

NUM_MODES = 25
PREDS_PER_MODE = 12

EPOCHS = 40 + WARMUP_EPOCHS
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
REPORT_INTERVAL = 15 # batches
PRINT_INTERVAL = 15 # batches

PICKLE_DATASET = False
PICKLE_DATASET_PATH = "simple_cnn_dataset.pickle"

CHUNK_DATASET = True
CHUNK_DATASET_PATH = "./chunks/"

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

# time this
start = time.time()
sample = d[0]
end = time.time()
print("=========")
print(f"Time to get sample: {end - start}", flush=True)
print(f"length of dataset: {len(d)}, estimated time to load all (mins): {(end - start) * len(d) / 60.0}")
print("=========")


run_id = random_id(10)
model = SimpleCNN(num_modes=NUM_MODES, predictions_per_mode=PREDS_PER_MODE, state_vector_size=sample[1].shape[0])
device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
is_cuda = device.type == "cuda"
model.to(device)

# compile model.forward to make it faster
# model_forward = torch.jit.trace(model.forward, example_inputs=(sample[0].to(device), sample[1].to(device)))

training_loader = torch.utils.data.DataLoader(d, batch_size=BATCH_SIZE, shuffle=True, pin_memory=is_cuda, num_workers=8, persistent_workers=True, prefetch_factor=5)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

tb_writer = SummaryWriter()

if PROFILE:
    model.forward = profiler(model.forward, "forward")
    loss_function = profiler(loss_function, "loss")
    optimizer.step = profiler(optimizer.step, "optstep")
if NSIGHT_PROFILE:
    model.forward = nsight_profiler(model.forward, "forward")
    loss_function = nsight_profiler(loss_function, "loss")
    optimizer.step = nsight_profiler(optimizer.step, "optstep")

torchdynamo.optimize()
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

def train_one_epoch(epoch_index, model, optimizer, dataloader, device, tb_writer):
    model.train()
    running_loss = 0
    running_l1_loss = 0
    running_minADE = 0
    running_minFDE = 0
    last_loss = 0

    for i, data in enumerate(dataloader):
        # SimpleCNNDataset returns shape (1, C, H, W), dataloader returns (B, 1, C, H, W)
        # squeeze to remove the 1 dimension

        for j, d_ in enumerate(data):
            data[j] = d_.cuda(non_blocking=True)

        data[0] = data[0].squeeze(1)

        if CHECK_NAN:
            for _, d_ in enumerate(data):
                if d_.isnan().any():
                    print("NAN FOUND")
                    breakpoint()

        optimizer.zero_grad()

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

        if i % REPORT_INTERVAL == REPORT_INTERVAL - 1:
            running_loss /= REPORT_INTERVAL - 1 # loss per batch
            running_l1_loss /= REPORT_INTERVAL - 1
            running_minADE /= REPORT_INTERVAL - 1
            running_minFDE /= REPORT_INTERVAL - 1

            tb_x = epoch_index * len(dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', running_loss, tb_x)
            tb_writer.add_scalar('Loss/l1_loss', running_l1_loss, tb_x)
            tb_writer.add_scalar('Metrics/minADE', running_minADE, tb_x)
            tb_writer.add_scalar('Metrics/minFDE', running_minFDE, tb_x)

            running_loss = 0.
            running_l1_loss = 0.
            running_minADE = 0.
            running_minFDE = 0.

if __name__ == "__main__":
    if MEMORY_PROFILE:
        torch.cuda.memory._record_memory_history()
    for x in range(1, EPOCHS + 1):
        if NSIGHT_PROFILE and x == WARMUP_EPOCHS + 1:
            startNsight()
        start = time.time()
        #profiler(train_one_epoch, "big_profile")(x, model, optimizer, training_loader, device, tb_writer)
        train_one_epoch(x, model, optimizer, training_loader, device, tb_writer)
        end = time.time()
        print(f"Epoch {x} took {end - start} seconds")


        torch.save({
            'epoch': x,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "NUM_MODES": NUM_MODES,
            "PREDS_PER_MODE": PREDS_PER_MODE,
            "BATCH_SIZE": BATCH_SIZE,
            "INIT_LEARNING_RATE": LEARNING_RATE,
            "INIT_MOMENTUM": MOMENTUM,
        }, f"runs/simple_cnn_r{run_id}_e{x}_{get_time_str()}.pt")
    if MEMORY_PROFILE:
        torch.cuda.memory._dump_snapshot("memory_profiler_dump.pickle")
    if NSIGHT_PROFILE:
        stopNsight()

