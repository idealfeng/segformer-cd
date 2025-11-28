import numpy as np
from PIL import Image

arr = np.array(Image.open("data/WHUCD/train/label/0_1.tif").convert("L"))
print(np.unique(arr))
