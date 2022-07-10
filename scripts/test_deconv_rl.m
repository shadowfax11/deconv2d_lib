DATA_DIR = './data/';
IMG_STR = 'cameraman.mat';
PSF_STR = 'psf_gaussian.mat'; 
NOISE_LVL = 0.01;

addpath ./src/methods_matlab/
addpath ./src/utils_matlab/

% load test scene
load([DATA_DIR, 'test_scenes/', IMG_STR]);

% load test psf
load([DATA_DIR, 'test_psfs/', PSF_STR]);

% render image
b_orig = conv2(x, h, 'same');
b = b_orig + NOISE_LVL*randn(size(x));
b = clip(b);
snr_render = calc_snr(b, b_orig);

x_est = f_richardsonlucy(h, b);
snr_recon = calc_snr(x_est, x);
ssim_recon = ssim(x_est, x);

figure; 
t = tiledlayout(2,2,'TileSpacing','Compact','Padding','Compact');
nexttile; imagesc(h); colorbar; title('PSF'); axis image;
nexttile; imagesc(x); colorbar; title('Original Image'); axis image;
nexttile; imagesc(b); colorbar; axis image; ... 
    title(sprintf('Rendered Image\nSNR=%3.3fdB',snr_render)); 
nexttile; imagesc(x_est); colorbar; axis image; ...
    title(sprintf('Scene Reconstruction\nSNR=%3.3fdB, SSIM=%.3f',snr_recon,ssim_recon));
