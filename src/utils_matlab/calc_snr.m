function snr = calc_snr(img,img_orig)
%CALC_SNR Calculates signal-to-noise ratio (in dB) of image given original image
%   img: Image
%   img_orig: Original/GT Image
snr = 10*log10(sum(abs(img_orig(:)).^2)/sum(abs(img(:)-img_orig(:)).^2));
end

