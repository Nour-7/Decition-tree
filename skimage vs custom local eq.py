"""
============================
Local Histogram Equalization
============================

This example enhances an image with low contrast, using a method called *local
histogram equalization*, which spreads out the most frequent intensity values
in an image.

The equalized image [1]_ has a roughly linear cumulative distribution function
for each pixel neighborhood.

The local version [2]_ of the histogram equalization emphasized every local
graylevel variations.

References
----------
.. [1] https://en.wikipedia.org/wiki/Histogram_equalization
.. [2] https://en.wikipedia.org/wiki/Adaptive_histogram_equalization

"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image

from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank


matplotlib.rcParams['font.size'] = 9


def histogram(img):
    hist2 = np.asarray([np.count_nonzero(img == i) for i in range(256)])
    return hist2


def cum_histogram(hist):

    cum_hist = hist.copy()
    for i in np.arange(1, 256):
        cum_hist[i] = cum_hist[i - 1] + cum_hist[i]
    return cum_hist


def local_filter(wind_op, img, mask_size):
    res = np.zeros(shape=img.shape)
    offset = mask_size // 2
    for i in range(offset, img.shape[0] - offset):
        for j in range(offset, img.shape[1] - offset):
            wind = img[i - offset: i + offset + 1, j - offset: j + offset + 1]
            res[i][j] = wind_op(wind)
    return res


def pad(img, size):
    res = np.zeros((2 * size + img.shape[0], 2 * size + img.shape[1]))
    res[size: size + img.shape[0], size: size + img.shape[1]] = img
    return res


def window_histEq(img):
    height = img.shape[0]
    center = height // 2
    pixels = height ** 2
    hist = histogram(img)
    cum_hist = cum_histogram(hist)
    c = (int)(img[center][center])
    center_pixel = np.floor(cum_hist[c] * 255.0 / pixels)
    return center_pixel/255


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins)
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')

    xmin, xmax = dtype_range[image.dtype.type]
    ax_hist.set_xlim(0, xmax)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')

    return ax_img, ax_hist, ax_cdf


# Load an example image
img = img_as_ubyte(data.moon())

# img = PIL.Image.open(
#     r'D:\Documents\6th tearm\ML\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0326(a)(embedded_square_noisy_512).tif')
# img = np.asarray(img)
# img.setflags(write=1)

# Global equalize
img_rescale = exposure.equalize_hist(img)

# Equalization
selem = disk(30)
img_eq = rank.equalize(img, selem=selem)

pd = pad(np.asarray(img), 30)
eq = local_filter(window_histEq, pd, 30)
eq = eq[30:eq.shape[0] - 29, 30: eq.shape[1] - 29]

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 3), dtype=np.object)
axes[0, 0] = plt.subplot(2, 3, 1)
axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])
axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])
axes[1, 0] = plt.subplot(2, 3, 4)
axes[1, 1] = plt.subplot(2, 3, 5)
axes[1, 2] = plt.subplot(2, 3, 6)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(eq, axes[:, 0])
ax_img.set_title('My Local Eq')
ax_hist.set_ylabel('Number of pixels')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 1])
ax_img.set_title('Local equalize')
ax_cdf.set_ylabel('Fraction of total intensity')


ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 2])
ax_img.set_title('Global equalise')


# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()
