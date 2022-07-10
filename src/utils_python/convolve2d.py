import os 
import numpy as np
import numpy.fft as fft

class Convolve2DFFT:
    def __init__(self, psf):
        self.psf = psf
        self.psf_sz = psf.shape

    def conv2d(self, x):
        """
        Performs 2D convolution using FFT-based computations (after padding, to avoid circular convolutions)
        x(numpy.ndarray): scene (2D array)
        """
        h, w = x.shape
        self.psf = np.pad(self.psf, ((int(h/2),int(h/2)), (int(w/2),int(w/2))), 'constant')
        H = fft.fft2(fft.fftshift(self.psf))
        x = np.pad(x, ((int(self.psf_sz[0]/2),int(self.psf_sz[0]/2)), \
            (int(self.psf_sz[1]/2),int(self.psf_sz[1]/2))), 'constant')
        X = fft.fft2(fft.fftshift(x))
        y = np.real(fft.ifftshift(fft.ifft2(H*X)))
        h_start = int(self.psf_sz[0]/2)
        h_end = h_start + h
        w_start = int(self.psf_sz[1]/2)
        w_end = w_start + w
        y = y[h_start:h_end, w_start:w_end]

        return y
