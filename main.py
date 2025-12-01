from PIL import Image
import numpy as np

p = r"data/ChangeDetectionDataset/Real/subset/test/label/00000.jpg"
a = np.array(Image.open(p).convert("L"))
print(np.unique(a)[:20], a.min(), a.max())
