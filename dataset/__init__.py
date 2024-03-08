from .nuscenes import NuScenesDataset
from .simple_cnn_dataset import SimpleCNNDataset

def test_dataset(d):
    sets = [set(), set(), set()]
    for data in d:
        sets[0].add(data[0].shape)
        sets[1].add(data[1].shape)
        sets[2].add(data[2].shape)
    print(sets)

