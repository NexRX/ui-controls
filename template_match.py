# Read images
# search_image = imread('__fixtures__/screen-2k.png', mode='L')  # Larger image
# template_image = imread('__fixtures__/btn.png', mode='L')  # Smaller template image

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from PIL import Image
from io import BytesIO

def load_and_convert_image(file_path):
    img = Image.open(file_path).convert('L')
    return np.array(img)

background_url = '__fixtures__/screen-2k.png'
template_url = '__fixtures__/btn.png'

template = load_and_convert_image(template_url)
background = load_and_convert_image(background_url)

# Get image dimensions
bx, by = background.shape[1], background.shape[0]
tx, ty = template.shape[1], template.shape[0]

# FFT
Ga = fftpack.fft2(background)
Gb = fftpack.fft2(template, shape=(by, bx))  # Specify the shape for the template FFT
cross_spectrum = (Ga * np.conj(Gb)) / np.abs(Ga * np.conj(Gb))
c = np.real(fftpack.ifft2(cross_spectrum))

# Find peak correlation
max_c = np.max(np.abs(c))
ypeak, xpeak = np.unravel_index(np.argmax(c), c.shape)

# Print the location of the peak
print(f"Peak correlation location: (x, y) = ({xpeak}, {ypeak})")

# Create a single figure with four subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Plot the FFT of the background (Ga)
axs[0, 0].imshow(np.log1p(np.abs(Ga)), cmap='gray')
axs[0, 0].set_title('FFT of Background (Ga)')
axs[0, 0].axis('off')

# Plot the FFT of the template (Gb)
axs[0, 1].imshow(np.log1p(np.abs(Gb)), cmap='gray')
axs[0, 1].set_title('FFT of Template (Gb)')
axs[0, 1].axis('off')

# Plot the correlation map
axs[1, 0].imshow(c, cmap='gray')
axs[1, 0].set_title('Correlation Map')
axs[1, 0].axis('off')

# Plot the background image with the rectangle for best match
axs[1, 1].imshow(background, cmap='gray')
rect = plt.Rectangle((xpeak, ypeak), tx, ty, edgecolor='r', facecolor='none')
axs[1, 1].add_patch(rect)
axs[1, 1].set_title('Best Match')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()