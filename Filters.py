import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image


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

# Special Filters


def Smoothing(img):
    return np.mean(img)


def Median(img):
    return np.median(img)


def Sharpening(img):
    res = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res += img[i][j] * laplacian[i][j]
    if res > 255:
        res = 255
    if res < 0:
        res = 0
    return res


def Sharpening2(img):
    res = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res += img[i][j] * laplacian[i][j]
    return res


# laplican matrix
laplacian = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

####
# GLOABAL MEAN - VAR
meanG = np.mean(img)
varG = np.var(img)


def window_stat(img):
    height = img.shape[0]
    center = height // 2
    mean = np.mean(img)
    var = np.var(img)
    x = (mean <= .4 * meanG and 0.02 * varG <= var and 0.4 * varG >= var)
    if(x):
        return img[center][center] * 4
    else:
        return img[center][center]


####
# # REFERENCE PICTURE
img_ref = PIL.Image.open(
    r'D:\Documents\6th tearm\ML\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0316(3)(third_from_top).tif')
img_ref = np.asarray(img_ref)
hist_ref = histogram(img_ref)
cum_hist_ref = cum_histogram(hist_ref)
prob_cum_hist_ref = cum_hist_ref / pixels_ref
height_ref = img_ref.shape[0]
width_ref = img_ref.shape[1]
pixels_ref = width_ref * height_ref


def window_match(img):
    height = img.shape[0]
    center = height // 2
    height = img.shape[0]
    width = img.shape[1]
    pixels = width * height
    hist = histogram(img)
    cum_hist = cum_histogram(hist)
    prob_cum_hist = cum_hist / pixels
    K = 256
    new_values = np.zeros((K))
    dif = [np.argmin([np.absolute(prob_cum_hist_ref - prob_cum_hist[i])])
           for i in range(256)]
    dif = np.asarray(dif)
    center_pixel = dif[(int)(img[center][center])]
    return center_pixel


img = PIL.Image.open(
    r'D:\Documents\6th tearm\ML\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0338(a)(blurry_moon).tif')
img = np.asarray(img)
pad = pad(img, 3)


x = local_filter(Sharpening, pad, 3)
y = local_filter(Sharpening2, pad, 3)
plt.subplot(331)
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.subplot(332)
plt.imshow(x, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.subplot(333)
plt.imshow(y, cmap=plt.get_cmap('gray'))
plt.axis('off')

plt.show()
