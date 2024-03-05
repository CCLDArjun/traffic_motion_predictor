import sys
sys.path.append("../")

import time
import functools

import torch
from torch.utils.tensorboard import SummaryWriter

from models.simple_cnn import SimpleCNN, loss_function
import dataset

torch.set_printoptions(sci_mode=False)
torch.autograd.set_detect_anomaly(True)

NUM_MODES = 5
PREDS_PER_MODE = 12

MIN_XY = 842
MAX_XY = 2733

MAX_VELOCITY = 20
MIN_VELOCITY = 0

MAX_ACCELERATION = 10
MIN_ACCELERATION = -10

MAX_HEADING_CHANGE_RATE = 0.42
MIN_HEADING_CHANGE_RATE = -0.26

def normalize(value, min, max):
    return (value - min) / (max - min)

class SimpleCNNDataset(dataset.NuScenesDataset):
    @functools.lru_cache(maxsize=10_000_000)
    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        state_vector = torch.tensor([
            normalize(data["velocity"], MIN_VELOCITY, MAX_VELOCITY),
            normalize(data["acceleration"], MIN_ACCELERATION, MAX_ACCELERATION),
            normalize(data["heading_change_rate"], MIN_HEADING_CHANGE_RATE, MAX_HEADING_CHANGE_RATE),
        ])

        state_vector[state_vector.isnan()] = 0

        xy_in = torch.flatten(torch.from_numpy(data["past"]["agent_xy_global"]))
        xy_in = normalize(xy_in, MIN_XY, MAX_XY)
        
        state_vector = torch.cat([state_vector, xy_in])
        agent_rast = torch.from_numpy(data["agent_rast"])

        return (
            agent_rast.unsqueeze(0).permute(0, 3, 1, 2).float(),
            state_vector.float(),
            normalize(torch.from_numpy(data["future"]["agent_xy_global"]).float(), MIN_XY, MAX_XY),
        )

d = SimpleCNNDataset("../data/sets/v1.0-mini")

def normalization_values():
    maxes = [0, 0, 0]
    mins = [0, 0, 0]
    max_xy = 0
    min_xy = 0
    i = 0
    for data in d:
        if i % 50 == 0:
            print(i)
        for i in range(3):
            maxes[i] = max(maxes[i], data[1][i])
            mins[i] = min(mins[i], data[1][i])

        max_xy = max(max_xy, data[1][3:].max())
        min_xy = min(max_xy, data[1][3:].min())

        max_xy = max(max_xy, data[2].max())
        min_xy = min(max_xy, data[2].min())

    print("MAXES: ", maxes)
    print("MINS: ", mins)
    print("XY: ", min_xy, max_xy)

# time this
start = time.time()
sample = d[0]
end = time.time()
print("=========")
print(f"Time to get sample: {end - start}", flush=True)
print(f"length of dataset: {len(d)}, estimated time to load all (mins): {(end - start) * len(d) / 60.0}")
print("=========")

def test_dataset(d):
    sets = [set(), set(), set()]
    for data in d:
        sets[0].add(data[0].shape)
        sets[1].add(data[1].shape)
        sets[2].add(data[2].shape)
    print(sets)
import cProfile

#cProfile.run('test_dataset(d)')

model = SimpleCNN(num_modes=NUM_MODES, predictions_per_mode=PREDS_PER_MODE, state_vector_size=sample[1].shape[0])

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
is_cuda = device.type == "cuda"
model.to(device)

# compile model.forward to make it faster
#model_forward = torch.jit.trace(model.forward, example_inputs=(sample[0].to(device), sample[1].to(device)))

training_loader = torch.utils.data.DataLoader(d, batch_size=2, shuffle=True, pin_memory=is_cuda)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

tb_writer = SummaryWriter()

def train_one_epoch(epoch_index, model, optimizer, dataloader, device, tb_writer):
    model.train()
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(dataloader):
        # SimpleCNNDataset returns shape (1, C, H, W), dataloader returns (B, 1, C, H, W)
        # squeeze to remove the 1 dimension
        data[0] = data[0].squeeze(1)

        for d_ in data:
            if d_.isnan().any():
                print("NAN FOUND")
                breakpoint()

        optimizer.zero_grad()
        y_pred = model.forward(data[0], data[1])
        loss = loss_function(*y_pred, data[2])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1 == 0:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

        print(f"Epoch {epoch_index}, Batch {i}, Loss: {loss.item()}")

if __name__ == "__main__":
    for x in range(1,100):
        print("EPOCH: ", x)
        with torch.autograd.detect_anomaly():
            train_one_epoch(x, model, optimizer, training_loader, device, tb_writer)

