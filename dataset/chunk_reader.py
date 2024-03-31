import os, os.path
import functools

import torch

class ChunkReader(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.len = len(os.listdir(self.path))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.len

        return torch.load(os.path.join(self.path, f"{idx}-{idx+1}.pt"))

