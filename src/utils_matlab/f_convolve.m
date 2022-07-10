function [b] = f_convolve(x,h)
%F_CONVOLVE Performs 2D convolution: (h * x) 
%   x, h can be single channel or multi-channel

pad_x = [floor(size(x,1)/2), floor(size(x,2)/2)];
pad_h = [floor(size(h,1)/2), floor(size(h,2)/2)];

h = padarray(h, [pad_x(1), pad_x(2)], 0, 'both');
x = padarray(x, [pad_h(1), pad_h(2)], 0, 'both');

assert(size(h,1)==size(x,1));
assert(size(h,2)==size(x,2));

H = fft2(ifftshift(ifftshift(h,1),2)); 
X = fft2(x);
B = H.*X;
b = real(ifft2(B));

if numel(size(b))==3
    b = b(pad_h(1)+1:end-pad_h(1), pad_h(2)+1:end-pad_h(2), :);
else
    b = b(pad_h(1)+1:end-pad_h(1), pad_h(2)+1:end-pad_h(2));
end

end

