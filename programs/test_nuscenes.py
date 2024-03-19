import sys
sys.path.append("..")

import dataset
from pprint import pprint
import matplotlib.pyplot as plt

d = dataset.NuScenesDataset("../data/sets/v1.0-mini")
sam = d[140]

img1 = sam['top_down_repr']
img2 = sam['agent_rast']

breakpoint()

# display both images in a single plot
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img1)
axes[1].imshow(img2)

plt.show()


