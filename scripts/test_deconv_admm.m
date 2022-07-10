DATA_DIR = './data/test_scenes/';
DATA_STR = 'cameraman.mat';
PSF_DIR = './data/test_psfs/';
PSF_STR = 'psf_double_gaussian.mat';
NOISE_LVL= 0.01;

addpath ./src/methods_matlab/
addpath ./src/utils_matlab/

% load test scene
load([DATA_DIR, DATA_STR]);

% load psf
load([PSF_DIR, PSF_STR]);

% render image
b_orig = conv2(x, h, 'same');
b = b_orig + NOISE_LVL*randn(size(x));
b = clip(b, [0, 1]);
snr_render = calc_snr(b, b_orig);

% ADMM optimization parameters
optim = {};
optim.lambda_L1 = 1e-3;
optim.lambda_TV = 1e-3;
optim.max_iters = 25;
[x_est, optim_info] = f_admm_ls_2dtv_l1(b, h, optim);
snr_recon = calc_snr(optim_info.crop2d(x_est), x);
ssim_recon = ssim(optim_info.crop2d(x_est), x);

figure; 
t = tiledlayout(2,2,'TileSpacing','Compact','Padding','Compact');
nexttile; imagesc(h); colorbar; title('PSF'); axis image;
nexttile; imagesc(x); colorbar; title('Scene Original'); axis image;
nexttile; imagesc(b); colorbar; axis image; ... 
    title(sprintf('Blurred Image\nSNR=%3.3fdB', snr_render)); 
nexttile; imagesc(x_est); colorbar; axis image; ...
    title(sprintf('Scene Reconstruction\nSNR=%3.3f dB, SSIM=%.3f',snr_recon,ssim_recon));