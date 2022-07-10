function [x] = f_richardsonlucy(h, b)
%F_RICHARDSONLUCY Uses the in-built RL deconv MATLAB function
%   h: PSF (2D array)
%   b: Image (2D array)
fprintf("Performing Richardson-Lucy (RL) deconvolution\n");
x = deconvlucy(b, h);
end

