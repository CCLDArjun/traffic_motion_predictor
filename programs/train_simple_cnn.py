import torch
from models.simple_cnn import SimpleCNN
import dataset

NUM_MODES = 5
PREDS_PER_MODE = 12

class SimpleCNNDataset(dataset.NuScenesDataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        state_vector = torch.tensor([
            data["velocity"],
            data["acceleration"],
            data["heading"],
            torch.flatten(torch.from_numpy(data["past"]["agent_xy_global"])),
        ])

        return (
            torch.from_numpy(data["agent_rast"]),
            state_vector,
            torch.from_numpy(data["future"]["agent_xy_global"]),
        )

d = SimpleCNNDataset("../data/sets/v1.0-mini")
model = SimpleCNN(num_modes=NUM_MODES, predictions_per_mode=PREDS_PER_MODE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_cuda = device.type == "cuda"
model.to(device)

# compile model.forward to make it faster
sample = d[0]
model_forward = torch.jit.trace(model.forward, sample[0].to(device), sample[1].to(device))

training_loader = torch.utils.data.DataLoader(d, batch_size=2, shuffle=True, pin_memory=is_cuda)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoc_index, model, optimizer, dataset, device):
    model.train()
    for i, data in enumerate(training_loader):
        optimizer.zero_grad()
        y_pred = model_forward(data["state_vector"], ))
        loss = torch.nn.functional.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoc_index}, Batch {batch_index}, Loss: {loss.item()}")

