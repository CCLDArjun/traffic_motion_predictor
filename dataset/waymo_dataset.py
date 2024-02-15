from torch.utils.data import Dataset


class WaymoDataset(Dataset):
    def __init__(self, dataroot):
        self.dataroot = dataroot

    def __len__(self):
        # TODO (not that important)
        return 0

    def __getitem__(self, idx):
        # TODO: return a sample, do the filtering here
        return data

