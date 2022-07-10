import numpy as np

def clip(x, clip_vals):
    y = x
    if not np.isnan(clip_vals[0]):
        y[np.where(x<clip_vals[0])] = clip_vals[0]
    if not np.isnan(clip_vals[1]):
        y[np.where(x>clip_vals[1])] = clip_vals[1]
    return y

def calc_snr(img, img_orig):
    snr = 10*np.log10(np.sum(np.abs(img_orig)**2)/(np.sum(np.abs(img - img_orig)**2)))
    return snr