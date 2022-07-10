function [x] = f_weiner_deconv(h, b, nsr)
%F_WEINER_DECONV Summary of this function goes here
%   h: PSF, 2D array
%   b: Blurred image, 2D array
%   nsr: Noise-to-signal ratio. Setting this to zero, would mean using the
%   ideal inverse filter

fprintf("Performing Weiner deconvolution\n");
if nargin<3
    nsr = 0;
    fprintf("Setting noise-to-signal raio=0, using ideal inverse filter\n");
end
x = deconvwnr(b, h, nsr);
end

