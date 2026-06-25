import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import decode_image


mask_example = decode_image(
    os.path.join(os.path.dirname(__file__), "trial.png")
).squeeze().numpy()
fig = plt.figure(figsize=(10, 10))

plt.imshow(mask_example, cmap='Spectral')
plt.title("Mask")
plt.axis('off')
plt.show()