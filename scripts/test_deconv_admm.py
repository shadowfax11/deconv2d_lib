import os, sys
import numpy as np
from scipy.io import loadmat
sys.path.append("../")
from src.utils_python.convolve2d import Convolve2DFFT
from src.utils_python.utils import clip, calc_snr
from src.methods_python import admm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

DATA_DIR = '../data/test_scenes/'
DATA_STR = 'cameraman.mat'
PSF_DIR = '../data/test_psfs/'
PSF_STR = 'psf_double_gaussian.mat'
NOISE_LVL = 0.01

# load test scene
img = loadmat(os.path.join(DATA_DIR, DATA_STR))
x = img['x']

# load test psf
h = loadmat(os.path.join(PSF_DIR, PSF_STR))
h = h['h']

# render out image
conv2d_obj = Convolve2DFFT(h)
b_orig = conv2d_obj.conv2d(x) 
b = b_orig + NOISE_LVL*np.random.randn(x.shape[0], x.shape[1])
b = clip(b, [0, 1])

# set ADMM params
optim = {}
optim['lambda_L1'] = 1e-3
optim['lambda_TV'] = 1e-3
optim['max_iters'] = 25

# run ADMM
x_est, optim_info, pad2d, crop2d = admm.admm_ls_2dtv_l1(b, h, optim)

# plot results
fh1 = plt.figure()
ax0 = plt.subplot(2,2,1)
im0 = ax0.imshow(h)
ax0.set_title('PSF')
fh1.colorbar(im0, ax=ax0)
ax1 = plt.subplot(2,2,2)
im1 = ax1.imshow(x)
ax1.set_title('Scene Original')
fh1.colorbar(im1, ax=ax1)
ax2 = plt.subplot(2,2,3)
im2 = ax2.imshow(b)
ax2.set_title('Image\nSNR={:2.3f}dB'.format(calc_snr(b, b_orig)))
fh1.colorbar(im2, ax=ax2)
ax3 = plt.subplot(2,2,4)
im3 = ax3.imshow(x_est)
ax3.set_title('Scene Reconstructed\nSNR={:2.3f}dB, SSIM={:.3f}'.format(calc_snr(x_est, pad2d(x)),ssim(crop2d(x_est), x)))
fh1.colorbar(im3, ax=ax3)
print("Image restored, SNR=%3.3f dB, SSIM=%2.3f"%(calc_snr(x_est, pad2d(x)),ssim(crop2d(x_est), x)))
plt.show()
