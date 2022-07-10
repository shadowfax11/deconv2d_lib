import os
import numpy as np
from skimage import restoration
from scipy.io import loadmat
from utils_python.convolve2d import Convolve2DFFT
from utils_python.utils import clip, calc_snr
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

DATA_DIR = './test_data/'
IMG_STR = 'cameraman.mat'
PSF_STR = 'psf_gaussian.mat'
NOISE_LVL = 0.01

img = loadmat(os.path.join(DATA_DIR, IMG_STR))
img = img['img']
h = loadmat(os.path.join(DATA_DIR, PSF_STR))
h = h['h']

conv2d_obj = Convolve2DFFT(h)
b = conv2d_obj.conv2d(img) 
b += NOISE_LVL*np.random.randn(img.shape[0], img.shape[1])
b = clip(b, [0, 1])

print("Performing Richardson-Lucy deconvolution")
x = restoration.richardson_lucy(b, h)

fh1 = plt.figure()
ax0 = plt.subplot(2,2,1)
im0 = ax0.imshow(h)
ax0.set_title('PSF')
fh1.colorbar(im0, ax=ax0)
ax1 = plt.subplot(2,2,2)
im1 = ax1.imshow(img)
ax1.set_title('Scene Original')
fh1.colorbar(im1, ax=ax1)
ax2 = plt.subplot(2,2,3)
im2 = ax2.imshow(b)
ax2.set_title('Image')
fh1.colorbar(im2, ax=ax2)
ax3 = plt.subplot(2,2,4)
im3 = ax3.imshow(x)
ax3.set_title('Scene Reconstructed\nSNR={:2.3f}dB'.format(calc_snr(x, img)))
fh1.colorbar(im3, ax=ax3)
plt.show()
print("Image restored, SNR=%3.3f dB, SSIM=%2.3f"%(calc_snr(x, img),ssim(x, img)))
