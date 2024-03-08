import sys
sys.path.append("../")

import time

import torch
from torch.utils.tensorboard import SummaryWriter

from models.simple_cnn import SimpleCNN, loss_function
from dataset import SimpleCNNDataset

torch.set_printoptions(sci_mode=False)
torch.autograd.set_detect_anomaly(True)

cpu = False

NUM_MODES = 5
PREDS_PER_MODE = 12

d = SimpleCNNDataset("../data/sets/v1.0-mini")

# time this
start = time.time()
sample = d[0]
end = time.time()
print("=========")
print(f"Time to get sample: {end - start}", flush=True)
print(f"length of dataset: {len(d)}, estimated time to load all (mins): {(end - start) * len(d) / 60.0}")
print("=========")

model = SimpleCNN(num_modes=NUM_MODES, predictions_per_mode=PREDS_PER_MODE, state_vector_size=sample[1].shape[0])

device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
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

        for i, d_ in enumerate(data):
            if d_.isnan().any():
                print("NAN FOUND")
                breakpoint()
            data[i] = d_.to(device)

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

