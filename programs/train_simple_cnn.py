import sys
sys.path.append("../")

import time

import torch
from torch.utils.tensorboard import SummaryWriter
from models.simple_cnn import SimpleCNN, loss_function
from torch.profiler import profile, record_function, ProfilerActivity

from dataset import SimpleCNNDataset
from utils import random_id, get_time_str

torch.set_printoptions(sci_mode=False)
torch.autograd.set_detect_anomaly(True)

cpu = False

NUM_MODES = 5
PREDS_PER_MODE = 12

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
MOMENTUM = 0.9

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

training_loader = torch.utils.data.DataLoader(d, batch_size=BATCH_SIZE, shuffle=True, pin_memory=is_cuda)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

tb_writer = SummaryWriter()


def train_one_epoch(epoch_index, model, optimizer, dataloader, device, tb_writer):
    model.train()
    running_loss = 0
    running_l1_loss = 0
    last_loss = 0

    for i, data in enumerate(dataloader):
        print(i)
        if i == 10: return
        # SimpleCNNDataset returns shape (1, C, H, W), dataloader returns (B, 1, C, H, W)
        # squeeze to remove the 1 dimension

        data[0] = data[0].squeeze(1)

        for i, d_ in enumerate(data):
            if d_.isnan().any():
                print("NAN FOUND")
                breakpoint()
            data[i] = d_.to(device)

        optimizer.zero_grad()
        y_pred = model.forward(data[0], data[1])
        loss, l1_loss = loss_function(*y_pred, data[2])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_l1_loss += l1_loss

        if i % 5 == 0:
            last_loss = running_loss / 5  # loss per batch
            running_l1_loss /= 5

            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.add_scalar('Loss/l1_loss', running_l1_loss, tb_x)

            running_loss = 0.
            running_l1_loss = 0.


if __name__ == "__main__":
    for x in range(1, 100):
        print("EPOCH: ", x)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
        ) as prof:
            train_one_epoch(x, model, optimizer, training_loader, device, tb_writer)
        prof.export_chrome_trace("trace.json")
        print("CPUUUU")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print("CUDAAA")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

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

