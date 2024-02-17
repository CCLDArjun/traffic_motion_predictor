import dataset
from pprint import pprint

d = dataset.NuScenesDataset("/Users/arjunbemarkar/Downloads/v1.0-mini/")
pprint(d[0])

