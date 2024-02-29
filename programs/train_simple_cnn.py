import sys
sys.path.append("../")

import torch
from models.simple_cnn import SimpleCNN, loss_function
import dataset
import time

NUM_MODES = 5
PREDS_PER_MODE = 12

class SimpleCNNDataset(dataset.NuScenesDataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        state_vector = torch.tensor([
            data["velocity"],
            data["acceleration"],
            data["heading_change_rate"],
        ])

        # TODO: state_vector differs in size from sample to sample, need to fix this
        state_vector = torch.cat([state_vector, torch.flatten(torch.from_numpy(data["past"]["agent_xy_global"]))])
        agent_rast = torch.from_numpy(data["agent_rast"])

        return (
            agent_rast.unsqueeze(0).permute(0, 3, 1, 2).float(),
            state_vector.float(),
            torch.from_numpy(data["future"]["agent_xy_global"]).float(),
        )

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

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
is_cuda = device.type == "cuda"
model.to(device)

# compile model.forward to make it faster
model_forward = torch.jit.trace(model.forward, example_inputs=(sample[0].to(device), sample[1].to(device)))

training_loader = torch.utils.data.DataLoader(d, batch_size=2, shuffle=True, pin_memory=is_cuda)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoc_index, model, optimizer, dataloader, device):
    model.train()
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        y_pred = model_forward(data[0], data[1])
        loss = loss_function(y_pred, data[2])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1 == 0:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

        print(f"Epoch {epoc_index}, Batch {batch_index}, Loss: {loss.item()}")

#train_one_epoch(0, model, optimizer, training_loader, device)

