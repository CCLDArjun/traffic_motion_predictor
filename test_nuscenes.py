import dataset
from pprint import pprint

d = dataset.NuScenesDataset("data/sets/nuscenes")
pprint(d[140])

