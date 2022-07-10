function [y] = clip(x,clip_vals)
%CLIP Clips the range of values of x to between [clip_vals(1),
%clip_vals(2)]. By default assumes to clip between [0,1]
%   x: Image/Scene (2D/3D array)
%   clip_vals: (2-element array). Min = clip_vals(1), Max = clip_vals(2)
%   Default is [0, 1]. If wishing to clip only in one direction, specify
%   the other element as NaN

if nargin<2
    clip_vals = [0, 1];
end

if sum(isnan(clip_vals))==0
    assert(clip_vals(1)<clip_vals(2));
end

y = x;
if ~isnan(clip_vals(1))
    y(x<clip_vals(1)) = clip_vals(1);
end
if ~isnan(clip_vals(2))
    y(x>clip_vals(2)) = clip_vals(2);
end
end

