from PIL import Image
import numpy as np
im = Image.open("b.png")
im.show()

print list(np.asarray(im))
